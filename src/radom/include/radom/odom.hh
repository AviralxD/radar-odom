#pragma once

#include <Eigen/Dense>
#include <optional>
#include <vector>

#include "radom/types.hh"

namespace radom {

struct VelocityEstimate {
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
  bool valid = false;
};

class OdometryEstimator {
public:
  OdometryEstimator() = default;

  std::optional<OdometryState> process(
      const RadarScan &scan,
      const std::vector<ImuMeasurement> &imu_batch);

private:
  std::vector<RadarPoint> sanitizeScan(
      const std::vector<RadarPoint> &raw,
      double r_min = 1.5,
      double r_max = 100.0,
      double iqr_k = 1.0);

  std::vector<RadarPoint> filterStationaryPoints(
      const std::vector<RadarPoint> &raw_points,
      double inlier_thresh_mps = 0.15);

  VelocityEstimate estimateEgoVelocity(
      const std::vector<RadarPoint> &inliers,
      double doppler_sign = -1.0,
      double lambda_z = 0.05,
      double huber_delta = 0.10,
      int irls_iters = 3);

  void integrateVelocity(const VelocityEstimate &vel_est, double dt);

  OdometryState state_;
  Eigen::Vector3d last_vel_world_ = Eigen::Vector3d::Zero();
  bool initialized_ = false;
  bool has_valid_vel_ = false;
  double last_scan_time_ = 0.0;
};

}