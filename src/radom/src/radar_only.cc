#include "radom/radar_only.hh"

namespace radom {

namespace {

// How quickly matched yaw corrections feed into the yaw rate: a correction is
// turned into a rate over this time rather than over the scan interval. Same
// gain as before at the nominal 20 Hz; but in some sequences (stream_02) two
// scans are occasionally stamped only ~7 ms apart, and dividing by that turned
// a 1 deg correction into a 70 deg/s rate, sending scan-to-scan into a spin.
constexpr double kYawRateTimeConstant = 0.1;  // [s]

}

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
  yaw_rate_ += yaw_correction / kYawRateTimeConstant;
  yaw_ = new_yaw;
  velocity_ = v_radar;
}

}
