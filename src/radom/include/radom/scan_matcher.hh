#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <deque>
#include <vector>

#include "radom/types.hh"

namespace radom {

// 2D (bird's eye) matching of a radar scan against recent keyframes.
// kScan matches against the previous scan only; kSubmap against the last
// `submap_size` keyframes.
enum class MatcherMode { kNone, kScan, kSubmap };

// What the match solves for: heading only, or x, y and heading
enum class MatcherDof { kYaw, kPlanar };

struct MatcherConfig {
  MatcherMode mode = MatcherMode::kNone;
  MatcherDof dof = MatcherDof::kPlanar;

  int submap_size = 20;              // keyframes kept in the submap
  double keyframe_distance = 0.5;    // [m] motion that triggers a new keyframe
  double keyframe_angle = 0.0873;    // [rad] rotation that triggers a new keyframe

  // Radar points are far less certain across the beam than along it, so each
  // point carries a covariance built from its polar measurement noise
  double sigma_range = 0.15;         // [m]
  double sigma_azimuth = 0.01745;    // [rad]

  double max_correspondence_distance = 2.0;  // [m]
  double robust_kernel = 2.0;        // Cauchy scale on the Mahalanobis distance
  double prior_yaw_std = 0.0349;     // [rad] trust in the predicted heading
  double prior_xy_std = 0.2;         // [m] trust in the predicted position

  int max_iterations = 10;
  int min_correspondences = 20;
  double max_yaw_correction = 0.0873;  // [rad] larger corrections are rejected
};

struct MatchResult {
  bool valid = false;
  double yaw = 0.0;                            // correction about world z [rad]
  Eigen::Vector2d xy = Eigen::Vector2d::Zero();  // correction in world x, y [m]
  int correspondences = 0;
};

class ScanMatcher {
public:
  explicit ScanMatcher(const MatcherConfig &cfg = MatcherConfig{}) : cfg_(cfg) {}

  bool enabled() const { return cfg_.mode != MatcherMode::kNone; }

  // Correction to `predicted` (radar pose in the world) that aligns the scan
  // with the submap. The rotation is about the vertical axis through the sensor.
  MatchResult match(const std::vector<RadarPoint> &points,
                    const Eigen::Isometry3d &predicted) const;

  // Insert the scan into the submap if it qualifies as a keyframe
  void addScan(const std::vector<RadarPoint> &points, const Eigen::Isometry3d &pose);

  void reset();

private:
  struct Cloud {
    Eigen::Matrix2Xd points;            // world x, y
    std::vector<Eigen::Matrix2d> covs;  // world frame
  };

  Cloud toWorld(const std::vector<RadarPoint> &points, const Eigen::Isometry3d &pose) const;

  MatcherConfig cfg_;
  std::deque<Cloud> keyframes_;
  Cloud submap_;  // all keyframes concatenated
  Eigen::Vector2d last_position_ = Eigen::Vector2d::Zero();
  double last_yaw_ = 0.0;
};

}
