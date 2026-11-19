#include "radom/odom.hh"

namespace radom {

std::optional<OdometryState> OdometryEstimator::process(
    const RadarScan &scan,
    const std::vector<ImuMeasurement> &imu_batch) {
  state_.stamp_ns = scan.stamp_ns;

  if (!initialized_) {
    last_scan_time_ = scan.timestamp;
    initialized_ = true;
    return state_;
  }

  const double dt = scan.timestamp - last_scan_time_;
  last_scan_time_ = scan.timestamp;
  if (dt <= 0.0) {
    return std::nullopt;
  }


  (void)imu_batch;

  return state_;
}

}