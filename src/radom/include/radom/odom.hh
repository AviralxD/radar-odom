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
  std::vector<RadarPoint> sanitizeScan(
    const std::vector<RadarPoint> &raw,
    double r_min = 1.5,
    double r_max = 100.0,
    double iqr_k = 1.0);

  std::vector<RadarPoint> filterStationaryPoints(
    const std::vector<RadarPoint> &raw_points,
    double inlier_thresh_mps = 0.15);

  OdometryState state_;
  bool initialized_ = false;
  double last_scan_time_ = 0.0;
};

}