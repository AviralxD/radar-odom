#include "radom/odom.hh"
#include "radom/ransac.hh"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace radom {

std::optional<OdometryState> OdometryEstimator::process(
    const RadarScan &scan,
    const std::vector<ImuMeasurement> &imu_batch) 
{
  // Detect rosbag loop / backward time jump and reset filter state cleanly
  if (has_last_scan_ && scan.timestamp < last_scan_time_) {
    rio_filter_.reset();
    state_ = OdometryState{};
    has_last_imu_ = false;
    has_last_scan_ = false;
  }
  last_scan_time_ = scan.timestamp;
  has_last_scan_ = true;
  state_.stamp_ns = scan.stamp_ns;

  // 0. Static IMU alignment warmup
  if (!rio_filter_.isInitialized()) {
    if (!rio_filter_.initializeFromStaticImu(imu_batch)) {
      return std::nullopt;
    }
    if (!imu_batch.empty()) {
      last_imu_time_ = imu_batch.back().timestamp;
      last_gyro_raw_ = imu_batch.back().angular_vel;
      last_accel_raw_ = imu_batch.back().linear_accel;
      has_last_imu_ = true;
    }
    return std::nullopt;
  }

  // 1. High-rate IMU strap-down propagation across batch
  for (const auto &imu : imu_batch) {
    if (has_last_imu_) {
      const double dt = imu.timestamp - last_imu_time_;
      if (dt > 0.0 && dt < 0.5) {
        const Eigen::Vector3d gyro_mid =
            0.5 * (imu.angular_vel + last_gyro_raw_);
        const Eigen::Vector3d accel_mid =
            0.5 * (imu.linear_accel + last_accel_raw_);
        ImuMeasurement midpoint = imu;
        midpoint.angular_vel = gyro_mid;
        midpoint.linear_accel = accel_mid;
        rio_filter_.propagateImu(midpoint, dt);
      }
    }
    last_imu_time_ = imu.timestamp;
    last_gyro_raw_ = imu.angular_vel;
    last_accel_raw_ = imu.linear_accel;
    has_last_imu_ = true;
  }

  // 2. RANSAC or GNC stationary point filtering + IRLS ego-velocity estimation
  const auto inliers = filterStationaryPoints(scan.points);
  const auto vel_est = estimateEgoVelocity(inliers);
  stats_.velocity = vel_est;

  // 3. RIEKF measurement update with lever-arm compensation
  last_update_ = UpdateResult::kInvalid;
  if (vel_est.valid) {
    last_update_ = rio_filter_.updateRadarVelocity(
        vel_est.velocity, vel_est.covariance, last_gyro_raw_);
  }

  // 4. Populate OdometryState in the 4D radar frame for publishing & gt_radar.tum eval
  state_.position = rio_filter_.radarPosition();
  state_.orientation = rio_filter_.radarOrientation();
  state_.linear_velocity = rio_filter_.radarLinearVelocity(last_gyro_raw_);
  state_.angular_velocity = rio_filter_.radarAngularVelocity(last_gyro_raw_);
  state_.pose_covariance = rio_filter_.radarPoseCovariance();
  state_.twist_covariance = rio_filter_.radarTwistCovariance();

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
  std::vector<float> rcs_sorted = rcs_vals;
  std::sort(rcs_sorted.begin(), rcs_sorted.end());
  const float q1 = rcs_sorted[q1_idx];
  const float q3 = rcs_sorted[q3_idx];

  const float min_rcs = q1 - static_cast<float>(iqr_k) * (q3 - q1);

  std::vector<RadarPoint> clean;
  clean.reserve(valid.size());
  for (const auto &pt : valid) {
    if (pt.rcs >= min_rcs) clean.push_back(pt);
  }
  return clean;
}

std::vector<RadarPoint> OdometryEstimator::filterStationaryPoints(
  const std::vector<RadarPoint> &raw_points)
{
  const auto clean_pts = sanitizeScan(raw_points);
  stats_ = FrontendStats{};
  stats_.num_points = static_cast<int>(clean_pts.size());
  if (clean_pts.size() < 4) return {};

  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  const auto start = std::chrono::steady_clock::now();
  const bool ok = frontend_.type == Frontend::kGnc
                      ? estimateGnc(clean_pts, &velocity)
                      : estimateRansac(clean_pts, &velocity);
  stats_.runtime_us = std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now() - start).count();
  if (!ok) return {};

  // Both front-ends classify inliers by the same rule
  const double sq_thresh = frontend_.inlier_thresh_mps * frontend_.inlier_thresh_mps;
  const RANSAC model(clean_pts, /*doppler_sign=*/-1.0);
  std::vector<RadarPoint> inliers;
  inliers.reserve(clean_pts.size());
  for (int i = 0; i < model.num_data(); ++i) {
    if (model.EvaluateModelOnPoint(velocity, i) < sq_thresh) {
      inliers.push_back(clean_pts[static_cast<size_t>(i)]);
    }
  }
  stats_.num_inliers = static_cast<int>(inliers.size());
  return inliers;
}

bool OdometryEstimator::estimateRansac(
  const std::vector<RadarPoint> &pts,
  Eigen::Vector3d *velocity)
{
  RANSAC solver(pts, /*doppler_sign=*/-1.0);

  ransac_lib::LORansacOptions options;
  options.min_num_iterations_ = 20;
  options.max_num_iterations_ = 250;
  options.squared_inlier_threshold_ =
      frontend_.inlier_thresh_mps * frontend_.inlier_thresh_mps;

  ransac_lib::LocallyOptimizedMSAC<
      Eigen::Vector3d,
      std::vector<Eigen::Vector3d>,
      RANSAC> lomsac;

  ransac_lib::RansacStatistics stats;
  const int num_inliers = lomsac.EstimateModel(options, solver, velocity, &stats);
  stats_.iterations = static_cast<int>(stats.num_iterations);
  return num_inliers >= 3;
}

bool OdometryEstimator::estimateGnc(
  const std::vector<RadarPoint> &pts,
  Eigen::Vector3d *velocity)
{
  GncOptions options;
  options.noise_bound = frontend_.inlier_thresh_mps;
  options.mu_step = frontend_.gnc_mu_step;
  options.max_iterations = frontend_.gnc_max_iterations;

  const GncResult result = solveGncTls(pts, /*doppler_sign=*/-1.0, options);
  stats_.iterations = result.iterations;
  *velocity = result.velocity;
  return result.valid;
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