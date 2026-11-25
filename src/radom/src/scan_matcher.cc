#include "radom/scan_matcher.hh"

#include <algorithm>
#include <cmath>

namespace radom {

ScanMatcher::Cloud ScanMatcher::toWorld(const std::vector<RadarPoint> &points,
                                        const Eigen::Isometry3d &pose) const {
  Cloud cloud;
  cloud.points.resize(2, static_cast<Eigen::Index>(points.size()));
  cloud.covs.reserve(points.size());
  const Eigen::Matrix2d R_heading =
      Eigen::Rotation2Dd(headingOf(pose.rotation())).toRotationMatrix();

  for (size_t i = 0; i < points.size(); ++i) {
    const Eigen::Vector3d p(points[i].x, points[i].y, points[i].z);
    cloud.points.col(static_cast<Eigen::Index>(i)) = (pose * p).head<2>();

    // Polar noise in the sensor frame: along the beam, and across it
    const double range = std::hypot(p.x(), p.y());
    const Eigen::Matrix2d R_az =
        Eigen::Rotation2Dd(std::atan2(p.y(), p.x())).toRotationMatrix();
    const double across = range * cfg_.sigma_azimuth;
    const Eigen::Vector2d variance(
        cfg_.sigma_range * cfg_.sigma_range,
        across * across + cfg_.sigma_range * cfg_.sigma_range);
    const Eigen::Matrix2d R = R_heading * R_az;
    cloud.covs.push_back(R * variance.asDiagonal() * R.transpose());
  }
  return cloud;
}

MatchResult ScanMatcher::match(const std::vector<RadarPoint> &points,
                               const Eigen::Isometry3d &predicted) const {
  MatchResult result;
  const Eigen::Index n_map = submap_.points.cols();
  if (!enabled() || n_map == 0 ||
      static_cast<int>(points.size()) < cfg_.min_correspondences) {
    return result;
  }

  const Cloud scan = toWorld(points, predicted);
  const Eigen::Vector2d origin = predicted.translation().head<2>();
  const bool planar = cfg_.dof == MatcherDof::kPlanar;
  const double max_sq = cfg_.max_correspondence_distance * cfg_.max_correspondence_distance;
  const double kernel_sq = cfg_.robust_kernel * cfg_.robust_kernel;

  // State [yaw, x, y]; the prior pulls it towards the predicted pose
  Eigen::Vector3d x = Eigen::Vector3d::Zero();
  const Eigen::Vector3d prior_info(
      1.0 / (cfg_.prior_yaw_std * cfg_.prior_yaw_std),
      1.0 / (cfg_.prior_xy_std * cfg_.prior_xy_std),
      1.0 / (cfg_.prior_xy_std * cfg_.prior_xy_std));

  Eigen::VectorXd sq_dist(n_map);
  for (int iter = 0; iter < cfg_.max_iterations; ++iter) {
    const Eigen::Matrix2d R = Eigen::Rotation2Dd(x(0)).toRotationMatrix();
    Eigen::Matrix3d H = prior_info.asDiagonal();
    Eigen::Vector3d g = prior_info.cwiseProduct(x);
    int count = 0;

    for (Eigen::Index i = 0; i < scan.points.cols(); ++i) {
      const Eigen::Vector2d arm = R * (scan.points.col(i) - origin);
      const Eigen::Vector2d q = arm + origin + x.tail<2>();

      Eigen::Index nearest = 0;
      sq_dist = (submap_.points.colwise() - q).colwise().squaredNorm().transpose();
      if (sq_dist.minCoeff(&nearest) > max_sq) continue;

      const Eigen::Vector2d e = q - submap_.points.col(nearest);
      const Eigen::Matrix2d cov =
          R * scan.covs[static_cast<size_t>(i)] * R.transpose() +
          submap_.covs[static_cast<size_t>(nearest)];
      const Eigen::Matrix2d info = cov.inverse();
      const double weight = 1.0 / (1.0 + e.dot(info * e) / kernel_sq);

      Eigen::Matrix<double, 2, 3> J;
      J.col(0) << -arm.y(), arm.x();
      J.rightCols<2>().setIdentity();

      H.noalias() += weight * J.transpose() * info * J;
      g.noalias() += weight * J.transpose() * info * e;
      ++count;
    }

    result.correspondences = count;
    if (count < cfg_.min_correspondences) return result;

    Eigen::Vector3d step = Eigen::Vector3d::Zero();
    if (planar) {
      step = -H.ldlt().solve(g);
    } else {
      step(0) = -g(0) / H(0, 0);
    }
    x += step;
    if (step.norm() < 1e-4) break;
  }

  if (!x.allFinite() || std::abs(x(0)) > cfg_.max_yaw_correction) return result;
  result.yaw = x(0);
  result.xy = x.tail<2>();
  result.valid = true;
  return result;
}

void ScanMatcher::addScan(const std::vector<RadarPoint> &points,
                          const Eigen::Isometry3d &pose) {
  if (!enabled() || points.empty()) return;

  const bool scan_mode = cfg_.mode == MatcherMode::kScan;
  const Eigen::Vector2d position = pose.translation().head<2>();
  const double yaw = headingOf(pose.rotation());
  if (!scan_mode && !keyframes_.empty() &&
      (position - last_position_).norm() < cfg_.keyframe_distance &&
      std::abs(wrapAngle(yaw - last_yaw_)) < cfg_.keyframe_angle) {
    return;
  }
  last_position_ = position;
  last_yaw_ = yaw;

  keyframes_.push_back(toWorld(points, pose));
  const size_t window = scan_mode ? 1 : static_cast<size_t>(std::max(cfg_.submap_size, 1));
  while (keyframes_.size() > window) keyframes_.pop_front();

  Eigen::Index total = 0;
  for (const auto &kf : keyframes_) total += kf.points.cols();
  submap_.points.resize(2, total);
  submap_.covs.clear();
  submap_.covs.reserve(static_cast<size_t>(total));
  Eigen::Index offset = 0;
  for (const auto &kf : keyframes_) {
    submap_.points.middleCols(offset, kf.points.cols()) = kf.points;
    submap_.covs.insert(submap_.covs.end(), kf.covs.begin(), kf.covs.end());
    offset += kf.points.cols();
  }
}

void ScanMatcher::reset() {
  keyframes_.clear();
  submap_ = Cloud{};
}

}
