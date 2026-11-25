#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <optional>
#include <vector>

#include "radom/frontend.hh"
#include "radom/radar_only.hh"
#include "radom/rio_iekf.hh"
#include "radom/scan_matcher.hh"
#include "radom/types.hh"

namespace radom {

// One odometry pipeline. The pieces are independent, so ablation tests are
// just different values of this struct:
//   front-end  (RANSAC | GNC)     stationary points and Doppler velocity
//   use_imu    (true | false)     IEKF with IMU, or radar-only dead reckoning
//   matcher    (none|scan|submap) heading from point cloud alignment
struct EstimatorConfig {
  bool use_imu = true;
  FrontendConfig frontend;
  MatcherConfig matcher;
  RioConfig rio;

  // With the IMU, the matched heading is a measurement into the IEKF
  double scan_yaw_std = 0.0349;   // [rad]
  double scan_yaw_gate = 10.83;   // NIS gate, 1 DoF chi-square 99.9%
};

// Diagnostics of the most recent process() call
struct EstimatorStats {
  FrontendStats frontend;
  VelocityEstimate velocity;  // radar velocity measurement from the front-end
  // Whether the velocity was used (radar-only runs: whether it was available)
  UpdateResult velocity_update = UpdateResult::kInvalid;
  MatchResult match;
  double match_runtime_us = 0.0;  // matching plus submap maintenance
  bool heading_used = false;      // matched heading changed the estimate
};

class OdometryEstimator {
public:
  explicit OdometryEstimator(const EstimatorConfig &cfg = EstimatorConfig{});

  // `imu_batch` is ignored by radar-only pipelines
  std::optional<OdometryState> process(
      const RadarScan &scan,
      const std::vector<ImuMeasurement> &imu_batch);

  const EstimatorStats &lastStats() const { return stats_; }

private:
  std::optional<OdometryState> processInertial(
      const RadarScan &scan, const std::vector<ImuMeasurement> &imu_batch,
      const FrontendResult &front);
  OdometryState processRadarOnly(const RadarScan &scan, const FrontendResult &front, double dt);

  Eigen::Isometry3d filterRadarPose() const;
  void reset();

  EstimatorConfig cfg_;
  RadarFrontend frontend_;
  ScanMatcher matcher_;
  RadarInertialFilter rio_filter_;
  RadarOnlyOdometry radar_only_;
  EstimatorStats stats_;

  Eigen::Vector3d last_gyro_raw_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_accel_raw_ = Eigen::Vector3d::Zero();
  double last_imu_time_ = 0.0;
  double last_scan_time_ = 0.0;
  bool has_last_imu_ = false;
  bool has_last_scan_ = false;
};

}
