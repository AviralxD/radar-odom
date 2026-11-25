#include "radom/radar_only.hh"

namespace radom {

Eigen::Isometry3d RadarOnlyOdometry::makePose(const Eigen::Vector2d &position, double yaw) {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  pose.translation() << position, 0.0;
  return pose;
}

Eigen::Vector2d RadarOnlyOdometry::integrate(const Eigen::Vector3d &v_radar, double dt,
                                             double new_yaw) const {
  const Eigen::Vector2d v_mean = 0.5 * (velocity_ + v_radar).head<2>();
  const Eigen::Rotation2Dd heading(0.5 * (yaw_ + new_yaw));
  return position_ + heading * v_mean * dt;
}

Eigen::Isometry3d RadarOnlyOdometry::predict(const Eigen::Vector3d &v_radar, double dt) const {
  const double yaw = yaw_ + yaw_rate_ * dt;
  return makePose(integrate(v_radar, dt, yaw), yaw);
}

void RadarOnlyOdometry::update(const Eigen::Vector3d &v_radar, double dt,
                               double yaw_correction, const Eigen::Vector2d &xy_correction) {
  const double new_yaw = yaw_ + yaw_rate_ * dt + yaw_correction;
  position_ = integrate(v_radar, dt, new_yaw) + xy_correction;
  if (dt > 0.0) {
    yaw_rate_ = 0.5 * yaw_rate_ + 0.5 * (new_yaw - yaw_) / dt;
  }
  yaw_ = new_yaw;
  velocity_ = v_radar;
}

}
