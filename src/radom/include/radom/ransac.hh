#pragma once

#include <RansacLib/ransac.h>
#include <Eigen/Dense>
#include <vector>

#include "radom/types.hh"

namespace radom {

class RANSAC {
public:
  RANSAC(const std::vector<RadarPoint> &pts, double doppler_sign = -1.0);

  int MinimalSolver(const std::vector<int> &sample, std::vector<Eigen::Vector3d> *models) const;
  int NonMinimalSolver(const std::vector<int> &sample, Eigen::Vector3d *model) const;

  int min_sample_size() const { return 3; }
  int non_minimal_sample_size() const { return 6; }
  int num_data() const { return static_cast<int>(U_.rows()); }

  double EvaluateModelOnPoint(const Eigen::Vector3d &model, int i) const {
    const double err = U_.row(i).dot(model) - y_(i);
    return err * err;
  }

  void LeastSquares(const std::vector<int> &sample, Eigen::Vector3d *model) const {
    NonMinimalSolver(sample, model);
  }

private:
  Eigen::MatrixX3d U_;
  Eigen::VectorXd y_;
};

}