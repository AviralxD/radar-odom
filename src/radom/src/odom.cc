#include "radom/odom.hh"
#include "radom/ransac.hh"

#include <algorithm>
#include <cmath>

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

  // 1. RANSAC Filtering of stationary RADAR points
  const auto inliers = filterStationaryPoints(scan.points);

  /* TODO:
   *   2. LSQ estimation of ego velocity
   *   3. Velocity integration to find position
   *   4. EKF with IMU data, integrate to find position
   *   5. GNC filtering instead of RANSAC
   */
  (void)inliers;
  (void)imu_batch;

  return state_;
}

std::vector<RadarPoint> OdometryEstimator::sanitizeScan(
  const std::vector<RadarPoint> &raw,
  double r_min,
  double r_max,
  double iqr_k) 
{
  std::vector<RadarPoint> valid;
  valid.reserve(raw.size());
  std::vector<float> rcs_vals;
  rcs_vals.reserve(raw.size());

  for (const auto &pt : raw) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z) ||
        !std::isfinite(pt.doppler) || !std::isfinite(pt.rcs)) {
      continue;
    }
    const double r = std::hypot(pt.x, pt.y, pt.z);
    if (r < r_min || r > r_max) continue;
    valid.push_back(pt);
    rcs_vals.push_back(pt.rcs);
  }

  if (rcs_vals.size() < 8) return valid;

  const size_t q1_idx = rcs_vals.size() / 4;
  const size_t q3_idx = (3 * rcs_vals.size()) / 4;
  std::nth_element(rcs_vals.begin(), rcs_vals.begin() + q1_idx, rcs_vals.end());
  const float q1 = rcs_vals[q1_idx];
  std::nth_element(rcs_vals.begin() + q1_idx, rcs_vals.begin() + q3_idx, rcs_vals.end());
  const float q3 = rcs_vals[q3_idx];

  const float min_rcs = q1 - static_cast<float>(iqr_k) * (q3 - q1);

  std::vector<RadarPoint> clean;
  clean.reserve(valid.size());
  for (const auto &pt : valid) {
    if (pt.rcs >= min_rcs) clean.push_back(pt);
  }
  return clean;
}

std::vector<RadarPoint> OdometryEstimator::filterStationaryPoints(
  const std::vector<RadarPoint> &raw_points,
  double inlier_thresh_mps) 
{
  const auto clean_pts = sanitizeScan(raw_points);
  if (clean_pts.size() < 4) return {};

  RANSAC solver(clean_pts, /*doppler_sign=*/-1.0);

  const double sq_thresh = inlier_thresh_mps * inlier_thresh_mps;
  ransac_lib::LORansacOptions options;
  options.min_num_iterations_ = 20;
  options.max_num_iterations_ = 250;
  options.squared_inlier_threshold_ = sq_thresh;

  ransac_lib::LocallyOptimizedMSAC<
      Eigen::Vector3d,
      std::vector<Eigen::Vector3d>,
      RANSAC> lomsac;

  Eigen::Vector3d best_vel = Eigen::Vector3d::Zero();
  ransac_lib::RansacStatistics stats;
  const int num_inliers = lomsac.EstimateModel(options, solver, &best_vel, &stats);
  if (num_inliers < 3) return {};

  std::vector<RadarPoint> inliers;
  inliers.reserve(static_cast<size_t>(num_inliers));
  for (int i = 0; i < solver.num_data(); ++i) {
    if (solver.EvaluateModelOnPoint(best_vel, i) < sq_thresh) {
      inliers.push_back(clean_pts[static_cast<size_t>(i)]);
    }
  }
  return inliers;
}

}