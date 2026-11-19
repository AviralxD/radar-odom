#include "radom/ransac.hh"

#include <cmath>

namespace radom {

RANSAC::RANSAC(const std::vector<RadarPoint> &pts, double doppler_sign)
    : U_(pts.size(), 3), y_(pts.size()) {
  for (size_t i = 0; i < pts.size(); ++i) {
    const Eigen::Vector3d p(pts[i].x, pts[i].y, pts[i].z);
    const double norm = p.norm();
		if (norm > 1e-6) {
      U_.row(i) = (p / norm).transpose();
    } else {
      U_.row(i).setZero();
    }    
		y_(i) = doppler_sign * static_cast<double>(pts[i].doppler);
  }
}

int RANSAC::MinimalSolver(const std::vector<int> &sample, std::vector<Eigen::Vector3d> *models) const {
  models->clear();
  Eigen::Matrix3d A;
  Eigen::Vector3d b;
  for (int i = 0; i < 3; ++i) {
    A.row(i) = U_.row(sample[i]);
    b(i) = y_(sample[i]);
  }
  auto qr = A.colPivHouseholderQr();
  if (qr.rank() < 3 || std::abs(A.determinant()) < 1e-4) return 0;

  models->push_back(qr.solve(b));
  return 1;
}

int RANSAC::NonMinimalSolver(const std::vector<int> &sample, Eigen::Vector3d *model) const {
  const int n = static_cast<int>(sample.size());
  if (n < 3) return 0;
  Eigen::MatrixX3d A(n, 3);
  Eigen::VectorXd b(n);
  for (int i = 0; i < n; ++i) {
    A.row(i) = U_.row(sample[i]);
    b(i) = y_(sample[i]);
  }
  auto qr = A.colPivHouseholderQr();
  if (qr.rank() < 3) return 0;
  *model = qr.solve(b);
  return 1;
}

}