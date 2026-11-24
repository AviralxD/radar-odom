#include "radom/odom.hh"
#include "radom/ransac.hh"

#include <algorithm>
#include <cmath>

namespace radom {

std::optional<OdometryState> OdometryEstimator::process(
    const RadarScan &scan,
    const std::vector<ImuMeasurement> &imu_batch) {
  (void)imu_batch;
  state_.stamp_ns = scan.stamp_ns;

  // 1. RANSAC Filtering of stationary RADAR points
  const auto inliers = filterStationaryPoints(scan.points);

  // 2. LSQ estimation of ego velocity
  const auto vel_est = estimateEgoVelocity(inliers);

  // 3. Velocity integration to find position
  if (!initialized_) {
    last_scan_time_ = scan.timestamp;
    initialized_ = true;
    integrateVelocity(vel_est, 0.0);
    return state_;
  }

  const double dt = scan.timestamp - last_scan_time_;
  last_scan_time_ = scan.timestamp;
  if (dt <= 0.0 || dt > 1.0) {
    has_valid_vel_ = false;
    return std::nullopt;
  }

  integrateVelocity(vel_est, dt);

  // TODO:
  // 4. Kalman filtering with IMU data
  // 5. GNC outlier rejection
  (void)inliers;
  (void)imu_batch;

  return state_;
}

std::vector<RadarPoint> OdometryEstimator::sanitizeScan(
  const std::vector<RadarPoint> &raw,
  double r_min,
  double r_max,
  double iqr_k) 
{
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
    if (r < r_min || r > r_max) continue;
    valid.push_back(pt);
    rcs_vals.push_back(pt.rcs);
  }

  if (rcs_vals.size() < 8) return valid;

  const size_t q1_idx = rcs_vals.size() / 4;
  const size_t q3_idx = (3 * rcs_vals.size()) / 4;
  std::nth_element(rcs_vals.begin(), rcs_vals.begin() + q1_idx, rcs_vals.end());
  const float q1 = rcs_vals[q1_idx];
  std::nth_element(rcs_vals.begin() + q1_idx, rcs_vals.begin() + q3_idx, rcs_vals.end());
  const float q3 = rcs_vals[q3_idx];

  const float min_rcs = q1 - static_cast<float>(iqr_k) * (q3 - q1);

  std::vector<RadarPoint> clean;
  clean.reserve(valid.size());
  for (const auto &pt : valid) {
    if (pt.rcs >= min_rcs) clean.push_back(pt);
  }
  return clean;
}

std::vector<RadarPoint> OdometryEstimator::filterStationaryPoints(
  const std::vector<RadarPoint> &raw_points,
  double inlier_thresh_mps) 
{
  const auto clean_pts = sanitizeScan(raw_points);
  if (clean_pts.size() < 4) return {};

  RANSAC solver(clean_pts, /*doppler_sign=*/-1.0);

  const double sq_thresh = inlier_thresh_mps * inlier_thresh_mps;
  ransac_lib::LORansacOptions options;
  options.min_num_iterations_ = 20;
  options.max_num_iterations_ = 250;
  options.squared_inlier_threshold_ = sq_thresh;

  ransac_lib::LocallyOptimizedMSAC<
      Eigen::Vector3d,
      std::vector<Eigen::Vector3d>,
      RANSAC> lomsac;

  Eigen::Vector3d best_vel = Eigen::Vector3d::Zero();
  ransac_lib::RansacStatistics stats;
  const int num_inliers = lomsac.EstimateModel(options, solver, &best_vel, &stats);
  if (num_inliers < 3) return {};

  std::vector<RadarPoint> inliers;
  inliers.reserve(static_cast<size_t>(num_inliers));
  for (int i = 0; i < solver.num_data(); ++i) {
    if (solver.EvaluateModelOnPoint(best_vel, i) < sq_thresh) {
      inliers.push_back(clean_pts[static_cast<size_t>(i)]);
    }
  }
  return inliers;
}

VelocityEstimate OdometryEstimator::estimateEgoVelocity(
  const std::vector<RadarPoint> &inliers,
  double doppler_sign,
  double lambda_z,
  double huber_delta,
  int irls_iters) 
{
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
    y(i) = doppler_sign * static_cast<double>(inliers[i].doppler);
  }

  Eigen::VectorXd weights = Eigen::VectorXd::Ones(n);
  const Eigen::Matrix3d reg = Eigen::Vector3d(1e-6, 1e-6, lambda_z).asDiagonal();

  Eigen::LDLT<Eigen::Matrix3d> ldlt;
  Eigen::VectorXd residuals(n);

  for (int iter = 0; iter <= irls_iters; ++iter) {
    const Eigen::Matrix3d H = U.transpose() * weights.asDiagonal() * U + reg;
    const Eigen::Vector3d b = U.transpose() * (weights.array() * y.array()).matrix();

    ldlt.compute(H);
    if (ldlt.info() != Eigen::Success) return est;

    est.velocity = ldlt.solve(b);
    residuals = U * est.velocity - y;

    if (iter < irls_iters) {
      for (int i = 0; i < n; ++i) {
        const double abs_r = std::abs(residuals(i));
        weights(i) = (abs_r <= huber_delta) ? 1.0 : (huber_delta / abs_r);
      }
    }
  }

  const double dof = static_cast<double>(std::max(1, n - 3));
  const double sigma2 = (residuals.array().square() * weights.array()).sum() / dof;
  est.covariance = std::max(sigma2, 1e-6) * ldlt.solve(Eigen::Matrix3d::Identity());
  est.valid = true;
  return est;
}

void OdometryEstimator::integrateVelocity(
    const VelocityEstimate &vel_est,
    double dt) 
{
  const Eigen::Matrix3d R_wb = state_.orientation.normalized().toRotationMatrix();

  if (vel_est.valid) {
    state_.linear_velocity = vel_est.velocity;
    state_.twist_covariance.topLeftCorner<3, 3>() = vel_est.covariance;

    const Eigen::Vector3d vel_world = R_wb * vel_est.velocity;
    if (has_valid_vel_) {
      state_.position += 0.5 * (last_vel_world_ + vel_world) * dt;
    } else {
      state_.position += vel_world * dt;
    }
    state_.pose_covariance.topLeftCorner<3, 3>() +=
        R_wb * vel_est.covariance * R_wb.transpose() * (dt * dt);

    last_vel_world_ = vel_world;
    has_valid_vel_ = true;
  } else if (has_valid_vel_) {
    state_.position += last_vel_world_ * dt;
    const Eigen::Matrix3d cov_body = state_.twist_covariance.topLeftCorner<3, 3>();
    state_.pose_covariance.topLeftCorner<3, 3>() +=
        R_wb * cov_body * R_wb.transpose() * (dt * dt);
  }
}

}