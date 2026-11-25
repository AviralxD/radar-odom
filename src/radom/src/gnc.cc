#include "radom/gnc.hh"

#include <cmath>

namespace radom {

GncResult solveGncTls(const std::vector<RadarPoint> &pts,
                      double doppler_sign,
                      const GncOptions &options) {
  GncResult result;
  const int n = static_cast<int>(pts.size());
  if (n < 3) return result;

  Eigen::MatrixX3d U(n, 3);
  Eigen::VectorXd y(n);
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3d p(pts[i].x, pts[i].y, pts[i].z);
    const double norm = p.norm();
    if (norm > 1e-6) {
      U.row(i) = (p / norm).transpose();
    } else {
      U.row(i).setZero();
    }
    y(i) = doppler_sign * static_cast<double>(pts[i].doppler);
  }

  Eigen::VectorXd weights = Eigen::VectorXd::Ones(n);
  Eigen::ArrayXd sq_residuals(n);

  const auto solveWeighted = [&]() {
    const Eigen::Matrix3d H = U.transpose() * weights.asDiagonal() * U +
                              options.regularization * Eigen::Matrix3d::Identity();
    const Eigen::Vector3d b = U.transpose() * weights.cwiseProduct(y);
    result.velocity = H.ldlt().solve(b);
    sq_residuals = (U * result.velocity - y).array().square();
    ++result.iterations;
    return result.velocity.allFinite();
  };

  // Start from the convex problem: ordinary least squares over all points
  if (!solveWeighted()) return result;

  const double c = options.noise_bound;
  const double c2 = c * c;
  const double denom = 2.0 * sq_residuals.maxCoeff() - c2;
  if (denom <= 0.0) {  // every point is already well inside the bound
    result.converged = true;
    result.valid = true;
    return result;
  }
  double mu = c2 / denom;

  while (result.iterations < options.max_iterations) {
    // Closed-form TLS weight update for the current surrogate
    const double lower = mu / (mu + 1.0) * c2;
    const double upper = (mu + 1.0) / mu * c2;
    bool binary = true;
    bool changed = false;
    int support = 0;
    for (int i = 0; i < n; ++i) {
      double w = 0.0;
      if (sq_residuals(i) <= lower) {
        w = 1.0;
      } else if (sq_residuals(i) < upper) {
        w = c * std::sqrt(mu * (mu + 1.0) / sq_residuals(i)) - mu;
        binary = false;
      }
      if (w != weights(i)) changed = true;
      if (w > 0.0) ++support;
      weights(i) = w;
    }
    if (support < 3) return result;

    // Same binary inlier set as last time, so the solution cannot move
    if (binary && !changed) {
      result.converged = true;
      break;
    }
    if (!solveWeighted()) return result;
    mu *= options.mu_step;
  }

  result.valid = true;
  return result;
}

}
