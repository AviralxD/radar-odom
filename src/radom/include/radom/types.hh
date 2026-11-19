#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cstdint>
#include <vector>

namespace radom {

struct RadarPoint {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  float doppler = 0.0f;
  float rcs = 0.0f;
};

struct RadarScan {
  int64_t stamp_ns = 0;    // ROS timestamp in ns
  double timestamp = 0.0;  // timestamp in s
  std::vector<RadarPoint> points;
};

struct ImuMeasurement {
  int64_t stamp_ns = 0;
  double timestamp = 0.0;
  Eigen::Vector3d linear_accel = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_vel = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

struct OdometryState {
  int64_t stamp_ns = 0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d linear_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 6, 6> pose_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-3;
  Eigen::Matrix<double, 6, 6> twist_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-3;
};

}