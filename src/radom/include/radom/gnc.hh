#pragma once

#include <Eigen/Dense>
#include <vector>

#include "radom/types.hh"

namespace radom {

// Graduated Non-Convexity with a Truncated Least Squares cost (GNC-TLS) for the
// radar Doppler model u_i^T v = y_i.
// H. Yang, P. Antonante, V. Tzoumas, L. Carlone, "Graduated Non-Convexity for
// Robust Spatial Perception", IEEE RA-L, 2020.
struct GncOptions {
  double noise_bound = 0.15;   // inlier residual bound [m/s]
  double mu_step = 1.4;        // growth of the convexity parameter per iteration
  int max_iterations = 100;
  double regularization = 1e-6;
};

struct GncResult {
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  int iterations = 0;       // weighted least-squares solves
  bool converged = false;   // weights settled to 0 or 1
  bool valid = false;
};

GncResult solveGncTls(const std::vector<RadarPoint> &pts,
                      double doppler_sign,
                      const GncOptions &options);

}
