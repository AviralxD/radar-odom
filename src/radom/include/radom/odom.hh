#pragma once

#include <optional>
#include <vector>
#include "radom/types.hh"

namespace radom {

class OdometryEstimator {
public:
  OdometryEstimator() = default;

  std::optional<OdometryState> process(
      const RadarScan &scan,
      const std::vector<ImuMeasurement> &imu_batch);

private:
  OdometryState state_;
  bool initialized_ = false;
  double last_scan_time_ = 0.0;

};

}