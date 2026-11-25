#include "radom/odom.hh"

#include <chrono>

namespace radom {

namespace {

double microsecondsSince(const std::chrono::steady_clock::time_point &start) {
  return std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now() - start).count();
}

}

OdometryEstimator::OdometryEstimator(const EstimatorConfig &cfg)
    : cfg_(cfg),
      frontend_(cfg.frontend),
      matcher_(cfg.matcher),
      rio_filter_(cfg.rio) {}

void OdometryEstimator::reset() {
  rio_filter_.reset();
  radar_only_.reset();
  matcher_.reset();
  has_last_imu_ = false;
  has_last_scan_ = false;
}

std::optional<OdometryState> OdometryEstimator::process(
    const RadarScan &scan,
    const std::vector<ImuMeasurement> &imu_batch)
{
  // Detect rosbag loop / backward time jump and reset cleanly
  if (has_last_scan_ && scan.timestamp < last_scan_time_) {
    reset();
  }
  const double scan_dt = has_last_scan_ ? scan.timestamp - last_scan_time_ : 0.0;
  last_scan_time_ = scan.timestamp;
  has_last_scan_ = true;
  stats_ = EstimatorStats{};

  const bool initialising = cfg_.use_imu && !rio_filter_.isInitialized();
  const FrontendResult front = initialising ? FrontendResult{} : frontend_.process(scan.points);
  stats_.frontend = front.stats;
  stats_.velocity = front.velocity;

  if (!cfg_.use_imu) {
    return processRadarOnly(scan, front, scan_dt);
  }
  return processInertial(scan, imu_batch, front);
}

std::optional<OdometryState> OdometryEstimator::processInertial(
    const RadarScan &scan,
    const std::vector<ImuMeasurement> &imu_batch,
    const FrontendResult &front)
{
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

  // 2. RIEKF Doppler velocity update with lever-arm compensation
  if (front.velocity.valid) {
    stats_.velocity_update = rio_filter_.updateRadarVelocity(
        front.velocity.velocity, front.velocity.covariance, last_gyro_raw_);
  }

  // 3. Optional heading update from scan matching, starting at the filter pose
  if (matcher_.enabled()) {
    const auto start = std::chrono::steady_clock::now();
    stats_.match = matcher_.match(front.inliers, filterRadarPose());
    if (stats_.match.valid) {
      stats_.heading_used = rio_filter_.updateHeading(
          stats_.match.yaw, cfg_.scan_yaw_std, cfg_.scan_yaw_gate) == UpdateResult::kAccepted;
    }
    matcher_.addScan(front.inliers, filterRadarPose());
    stats_.match_runtime_us = microsecondsSince(start);
  }

  // 4. Publishable state in the 4D radar frame
  OdometryState state;
  state.stamp_ns = scan.stamp_ns;
  state.position = rio_filter_.radarPosition();
  state.orientation = rio_filter_.radarOrientation();
  state.linear_velocity = rio_filter_.radarLinearVelocity(last_gyro_raw_);
  state.angular_velocity = rio_filter_.radarAngularVelocity(last_gyro_raw_);
  state.pose_covariance = rio_filter_.radarPoseCovariance();
  state.twist_covariance = rio_filter_.radarTwistCovariance();
  return state;
}

OdometryState OdometryEstimator::processRadarOnly(
    const RadarScan &scan, const FrontendResult &front, double dt)
{
  // Without a usable velocity, coast on the previous one
  const Eigen::Vector3d velocity =
      front.velocity.valid ? front.velocity.velocity : radar_only_.velocity();
  stats_.velocity_update =
      front.velocity.valid ? UpdateResult::kAccepted : UpdateResult::kInvalid;

  if (matcher_.enabled()) {
    const auto start = std::chrono::steady_clock::now();
    stats_.match = matcher_.match(front.inliers, radar_only_.predict(velocity, dt));
    stats_.match_runtime_us = microsecondsSince(start);
  }
  // An invalid match leaves its corrections at zero
  radar_only_.update(velocity, dt, stats_.match.yaw, stats_.match.xy);
  stats_.heading_used = stats_.match.valid;
  if (matcher_.enabled()) {
    const auto start = std::chrono::steady_clock::now();
    matcher_.addScan(front.inliers, radar_only_.pose());
    stats_.match_runtime_us += microsecondsSince(start);
  }

  OdometryState state;
  state.stamp_ns = scan.stamp_ns;
  state.position = radar_only_.pose().translation();
  state.orientation = Eigen::Quaterniond(radar_only_.pose().rotation());
  state.linear_velocity = velocity;
  state.angular_velocity = Eigen::Vector3d(0.0, 0.0, radar_only_.yawRate());
  // All zero marks the covariance as unknown: there is no filter behind it
  state.pose_covariance.setZero();
  state.twist_covariance.setZero();
  return state;
}

Eigen::Isometry3d OdometryEstimator::filterRadarPose() const {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = rio_filter_.radarOrientation().toRotationMatrix();
  pose.translation() = rio_filter_.radarPosition();
  return pose;
}

}
