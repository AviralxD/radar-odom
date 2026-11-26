#include "radom/frontend.hh"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "radom/gnc.hh"
#include "radom/ransac.hh"

namespace radom {

namespace {
// Doppler is positive for approaching targets: y = -doppler = u^T v
constexpr double kDopplerSign = -1.0;
}

FrontendResult RadarFrontend::process(const std::vector<RadarPoint> &raw_points) const {
  FrontendResult result;
  const auto clean_pts = sanitizeScan(raw_points);
  result.stats.num_points = static_cast<int>(clean_pts.size());
  result.stats.injected_points = static_cast<int>(std::count_if(
      clean_pts.begin(), clean_pts.end(), [](const RadarPoint &p) { return p.injected; }));
  if (clean_pts.size() < 4) return result;

  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  const auto start = std::chrono::steady_clock::now();
  const bool ok = cfg_.type == Frontend::kGnc
                      ? estimateGnc(clean_pts, &velocity, &result.stats.iterations)
                      : estimateRansac(clean_pts, &velocity, &result.stats.iterations);
  result.stats.runtime_us = std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now() - start).count();
  if (!ok) return result;

  // Both estimators classify inliers by the same rule
  const double sq_thresh = cfg_.inlier_thresh_mps * cfg_.inlier_thresh_mps;
  const RANSAC model(clean_pts, kDopplerSign);
  result.inliers.reserve(clean_pts.size());
  for (int i = 0; i < model.num_data(); ++i) {
    if (model.EvaluateModelOnPoint(velocity, i) < sq_thresh) {
      result.inliers.push_back(clean_pts[static_cast<size_t>(i)]);
      result.stats.injected_inliers += clean_pts[static_cast<size_t>(i)].injected ? 1 : 0;
    }
  }
  result.stats.num_inliers = static_cast<int>(result.inliers.size());
  result.velocity = estimateEgoVelocity(result.inliers);
  return result;
}

std::vector<RadarPoint> RadarFrontend::sanitizeScan(const std::vector<RadarPoint> &raw) const {
  std::vector<RadarPoint> valid;
  valid.reserve(raw.size());
  std::vector<float> rcs_vals;
  rcs_vals.reserve(raw.size());

  for (const auto &pt : raw) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z) ||
        !std::isfinite(pt.doppler) || !std::isfinite(pt.rcs)) {
      continue;
    }
    const double r = std::hypot(pt.x, pt.y, pt.z);
    if (r < cfg_.range_min || r > cfg_.range_max) continue;
    valid.push_back(pt);
    rcs_vals.push_back(pt.rcs);
  }

  if (rcs_vals.size() < 8) return valid;

  const size_t q1_idx = rcs_vals.size() / 4;
  const size_t q3_idx = (3 * rcs_vals.size()) / 4;
  std::vector<float> rcs_sorted = rcs_vals;
  std::sort(rcs_sorted.begin(), rcs_sorted.end());
  const float q1 = rcs_sorted[q1_idx];
  const float q3 = rcs_sorted[q3_idx];

  const float min_rcs = q1 - static_cast<float>(cfg_.rcs_iqr_k) * (q3 - q1);

  std::vector<RadarPoint> clean;
  clean.reserve(valid.size());
  for (const auto &pt : valid) {
    if (pt.rcs >= min_rcs) clean.push_back(pt);
  }
  return clean;
}

bool RadarFrontend::estimateRansac(const std::vector<RadarPoint> &pts,
                                   Eigen::Vector3d *velocity, int *iterations) const {
  RANSAC solver(pts, kDopplerSign);

  ransac_lib::LORansacOptions options;
  options.min_num_iterations_ = 20;
  options.max_num_iterations_ = 250;
  options.squared_inlier_threshold_ = cfg_.inlier_thresh_mps * cfg_.inlier_thresh_mps;

  ransac_lib::LocallyOptimizedMSAC<
      Eigen::Vector3d,
      std::vector<Eigen::Vector3d>,
      RANSAC> lomsac;

  ransac_lib::RansacStatistics stats;
  const int num_inliers = lomsac.EstimateModel(options, solver, velocity, &stats);
  *iterations = static_cast<int>(stats.num_iterations);
  return num_inliers >= 3;
}

bool RadarFrontend::estimateGnc(const std::vector<RadarPoint> &pts,
                                Eigen::Vector3d *velocity, int *iterations) const {
  GncOptions options;
  options.noise_bound = cfg_.inlier_thresh_mps;
  options.mu_step = cfg_.gnc_mu_step;
  options.max_iterations = cfg_.gnc_max_iterations;

  const GncResult result = solveGncTls(pts, kDopplerSign, options);
  *iterations = result.iterations;
  *velocity = result.velocity;
  return result.valid;
}

VelocityEstimate RadarFrontend::estimateEgoVelocity(
    const std::vector<RadarPoint> &inliers) const {
  VelocityEstimate est;
  const int n = static_cast<int>(inliers.size());
  if (n < 3) return est;

  Eigen::MatrixX3d U(n, 3);
  Eigen::VectorXd y(n);
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3d p(inliers[i].x, inliers[i].y, inliers[i].z);
    const double norm = p.norm();
    if (norm <= 1e-6) return est;
    U.row(i) = (p / norm).transpose();
    y(i) = kDopplerSign * static_cast<double>(inliers[i].doppler);
  }

  Eigen::VectorXd weights = Eigen::VectorXd::Ones(n);
  const Eigen::Matrix3d reg = Eigen::Vector3d(1e-6, 1e-6, cfg_.lambda_z).asDiagonal();

  Eigen::LDLT<Eigen::Matrix3d> ldlt;
  Eigen::VectorXd residuals(n);

  for (int iter = 0; iter <= cfg_.irls_iterations; ++iter) {
    const Eigen::Matrix3d H = U.transpose() * weights.asDiagonal() * U + reg;
    const Eigen::Vector3d b = U.transpose() * (weights.array() * y.array()).matrix();

    ldlt.compute(H);
    if (ldlt.info() != Eigen::Success) return est;

    est.velocity = ldlt.solve(b);
    residuals = U * est.velocity - y;

    if (iter < cfg_.irls_iterations) {
      for (int i = 0; i < n; ++i) {
        const double abs_r = std::abs(residuals(i));
        weights(i) = (abs_r <= cfg_.huber_delta) ? 1.0 : (cfg_.huber_delta / abs_r);
      }
    }
  }

  const double dof = static_cast<double>(std::max(1, n - 3));
  const double sigma2 = (residuals.array().square() * weights.array()).sum() / dof;
  est.covariance = std::max(sigma2, 1e-6) * ldlt.solve(Eigen::Matrix3d::Identity());
  est.valid = true;
  return est;
}

}
