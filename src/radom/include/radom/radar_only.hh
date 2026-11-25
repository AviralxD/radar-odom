#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace radom {

// Planar dead reckoning from radar alone: translation from the Doppler
// velocity, heading from scan matching. Without an IMU, roll, pitch and height
// cannot be observed and are held at zero.
class RadarOnlyOdometry {
public:
  // Predicted pose after `dt`, assuming the previous yaw rate continues
  Eigen::Isometry3d predict(const Eigen::Vector3d &v_radar, double dt) const;

  // Commit the step, applying the scan matcher's corrections to the prediction
  void update(const Eigen::Vector3d &v_radar, double dt,
              double yaw_correction, const Eigen::Vector2d &xy_correction);

  Eigen::Isometry3d pose() const { return makePose(position_, yaw_); }
  const Eigen::Vector3d &velocity() const { return velocity_; }  // radar frame
  double yawRate() const { return yaw_rate_; }
  void reset() { *this = RadarOnlyOdometry{}; }

private:
  static Eigen::Isometry3d makePose(const Eigen::Vector2d &position, double yaw);
  // Position after moving at the mean velocity with the mean heading
  Eigen::Vector2d integrate(const Eigen::Vector3d &v_radar, double dt, double new_yaw) const;

  Eigen::Vector2d position_ = Eigen::Vector2d::Zero();
  Eigen::Vector3d velocity_ = Eigen::Vector3d::Zero();
  double yaw_ = 0.0;
  double yaw_rate_ = 0.0;
};

}
