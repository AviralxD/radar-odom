#pragma once

#include <Eigen/Dense>
#include <vector>

#include "radom/types.hh"

namespace radom {

struct VelocityEstimate {
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
  bool valid = false;
};

// How stationary points are separated from outliers
enum class Frontend { kRansac, kGnc };

struct FrontendConfig {
  Frontend type = Frontend::kRansac;
  double inlier_thresh_mps = 0.15;  // RANSAC threshold and GNC noise bound
  double gnc_mu_step = 1.4;
  int gnc_max_iterations = 100;

  double range_min = 1.5;    // [m]
  double range_max = 100.0;  // [m]
  double rcs_iqr_k = 1.0;    // drop points with RCS below Q1 - k * IQR
  double lambda_z = 0.05;    // regularisation of the vertical velocity
  double huber_delta = 0.10; // [m/s]
  int irls_iterations = 3;
};

struct FrontendStats {
  int num_points = 0;    // points entering the robust estimator
  int num_inliers = 0;
  int iterations = 0;    // RANSAC hypotheses, or GNC weighted solves
  double runtime_us = 0.0;  // robust estimator only
  // Synthetic outliers (outlier ablation) among the points and the inliers
  int injected_points = 0;
  int injected_inliers = 0;
};

struct FrontendResult {
  std::vector<RadarPoint> inliers;  // stationary points
  VelocityEstimate velocity;        // radar velocity in the radar frame
  FrontendStats stats;
};

// Radar scan -> stationary points and ego-velocity
class RadarFrontend {
public:
  explicit RadarFrontend(const FrontendConfig &cfg = FrontendConfig{}) : cfg_(cfg) {}

  FrontendResult process(const std::vector<RadarPoint> &raw_points) const;

private:
  std::vector<RadarPoint> sanitizeScan(const std::vector<RadarPoint> &raw) const;

  // Each fills `velocity` with the robust estimate and returns false on failure
  bool estimateRansac(const std::vector<RadarPoint> &pts, Eigen::Vector3d *velocity,
                      int *iterations) const;
  bool estimateGnc(const std::vector<RadarPoint> &pts, Eigen::Vector3d *velocity,
                   int *iterations) const;

  VelocityEstimate estimateEgoVelocity(const std::vector<RadarPoint> &inliers) const;

  FrontendConfig cfg_;
};

}
