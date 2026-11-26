#include "radom/radom_node.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <geometry_msgs/msg/transform_stamped.hpp>

namespace radom {

namespace {

// A stamp this far behind the previous one means the dataset restarted
constexpr int64_t kTimeJumpNs = 500'000'000;
constexpr size_t kMaxPendingScans = 100;
constexpr double kDegToRad = M_PI / 180.0;
// Optional per-point label from outlier_injector.py; non-zero marks a synthetic outlier
constexpr const char *kInjectedField = "injected";

using sensor_msgs::msg::PointField;

const PointField *findField(const sensor_msgs::msg::PointCloud2 &msg, const std::string &name) {
  for (const auto &f : msg.fields) {
    if (f.name == name) return &f;
  }
  return nullptr;
}

size_t typeSize(uint8_t datatype) {
  switch (datatype) {
    case PointField::INT8:
    case PointField::UINT8: return 1;
    case PointField::INT16:
    case PointField::UINT16: return 2;
    case PointField::INT32:
    case PointField::UINT32:
    case PointField::FLOAT32: return 4;
    case PointField::FLOAT64: return 8;
    default: return 0;
  }
}

template <typename T>
double readAs(const uint8_t *ptr) {
  T value;
  std::memcpy(&value, ptr, sizeof(T));
  return static_cast<double>(value);
}

double readValue(const uint8_t *point, const PointField &field) {
  const uint8_t *ptr = point + field.offset;
  switch (field.datatype) {
    case PointField::INT8: return readAs<int8_t>(ptr);
    case PointField::UINT8: return readAs<uint8_t>(ptr);
    case PointField::INT16: return readAs<int16_t>(ptr);
    case PointField::UINT16: return readAs<uint16_t>(ptr);
    case PointField::INT32: return readAs<int32_t>(ptr);
    case PointField::UINT32: return readAs<uint32_t>(ptr);
    case PointField::FLOAT32: return readAs<float>(ptr);
    case PointField::FLOAT64: return readAs<double>(ptr);
    default: return 0.0;
  }
}

bool readable(const sensor_msgs::msg::PointCloud2 &msg, const PointField *field) {
  return field && typeSize(field->datatype) > 0 &&
         field->offset + typeSize(field->datatype) <= msg.point_step;
}

}

RadomNode::RadomNode() : Node("radom") {
  const auto radar_topic = this->declare_parameter<std::string>("radar_topic", "/continental/points");
  const auto imu_topic = this->declare_parameter<std::string>("imu_topic", "/imu/data_raw");
  const auto output_ns = this->declare_parameter<std::string>("output_namespace", "/radom");
  odom_frame_ = this->declare_parameter<std::string>("odom_frame", "odom");
  base_frame_ = this->declare_parameter<std::string>("base_frame", "continental");
  publish_tf_ = this->declare_parameter<bool>("publish_tf", true);
  doppler_field_ = this->declare_parameter<std::string>("doppler_field", "v");
  rcs_field_ = this->declare_parameter<std::string>("rcs_field", "RCS");
  radar_time_offset_s_ = this->declare_parameter<double>("radar_time_offset_s", 0.0);

  const auto runs = this->declare_parameter<std::vector<std::string>>("runs", {"imu"});
  if (runs.empty()) {
    throw std::invalid_argument("`runs` must name at least one pipeline");
  }
  tf_run_ = this->declare_parameter<std::string>("tf_run", runs.front());

  // Settings without a prefix are the defaults shared by every run
  const EstimatorConfig defaults = declareEstimatorConfig("", EstimatorConfig{});
  for (const auto &name : runs) {
    Pipeline pipeline;
    pipeline.name = name;
    pipeline.config = declareEstimatorConfig(name + ".", defaults);
    pipeline.estimator = std::make_unique<OdometryEstimator>(pipeline.config);
    pipeline.odom_pub = this->create_publisher<nav_msgs::msg::Odometry>(
        output_ns + "/" + name + "/odom", 50);
    pipeline.stats_pub = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        output_ns + "/" + name + "/stats", 50);
    pipelines_.push_back(std::move(pipeline));
  }
  checkOverrides();

  // Deep queues: several pipelines may take longer than one radar period, and a
  // dropped message would change the results
  radar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      radar_topic, rclcpp::SensorDataQoS().keep_last(100),
      std::bind(&RadomNode::radarCallback, this, std::placeholders::_1));

  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, rclcpp::SensorDataQoS().keep_last(1000),
      std::bind(&RadomNode::imuCallback, this, std::placeholders::_1));

  if (publish_tf_) {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  }

  for (const auto &pipeline : pipelines_) {
    const EstimatorConfig &c = pipeline.config;
    const char *matcher = c.matcher.mode == MatcherMode::kNone ? "none"
                          : c.matcher.mode == MatcherMode::kScan ? "scan" : "submap";
    RCLCPP_INFO(this->get_logger(), "run '%s': frontend=%s use_imu=%s matcher=%s -> %s/%s/odom",
                pipeline.name.c_str(),
                c.frontend.type == Frontend::kGnc ? "gnc" : "ransac",
                c.use_imu ? "true" : "false", matcher,
                output_ns.c_str(), pipeline.name.c_str());
  }
}

double RadomNode::declareNumber(const std::string &name, double fallback) {
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.dynamic_typing = true;
  const auto value =
      this->declare_parameter(name, rclcpp::ParameterValue(fallback), descriptor);
  if (value.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
    return static_cast<double>(value.get<int64_t>());
  }
  if (value.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
    return value.get<double>();
  }
  throw std::invalid_argument("parameter '" + name + "' must be a number");
}

EstimatorConfig RadomNode::declareEstimatorConfig(const std::string &prefix,
                                                  const EstimatorConfig &d) {
  EstimatorConfig c = d;
  const auto full = [&](const char *key) {
    known_parameters_.insert(prefix + key);
    return prefix + key;
  };
  const auto real = [&](const char *key, double fallback) {
    return declareNumber(full(key), fallback);
  };
  const auto integer = [&](const char *key, int fallback) {
    return static_cast<int>(this->declare_parameter<int>(full(key), fallback));
  };
  const auto degrees = [&](const char *key, double fallback_rad) {
    return real(key, fallback_rad / kDegToRad) * kDegToRad;
  };
  // A string parameter restricted to the listed choices; returns its index
  const auto choice = [&](const char *key, const std::vector<std::string> &options,
                          size_t fallback) {
    const auto value = this->declare_parameter<std::string>(full(key), options[fallback]);
    for (size_t i = 0; i < options.size(); ++i) {
      if (value == options[i]) return i;
    }
    throw std::invalid_argument("parameter '" + prefix + key + "': unknown value '" + value + "'");
  };

  c.use_imu = this->declare_parameter<bool>(full("use_imu"), d.use_imu);

  c.frontend.type = static_cast<Frontend>(
      choice("frontend", {"ransac", "gnc"}, static_cast<size_t>(d.frontend.type)));
  c.frontend.inlier_thresh_mps = real("inlier_threshold_mps", d.frontend.inlier_thresh_mps);
  c.frontend.gnc_mu_step = real("gnc_mu_step", d.frontend.gnc_mu_step);
  c.frontend.gnc_max_iterations = integer("gnc_max_iterations", d.frontend.gnc_max_iterations);
  c.frontend.range_min = real("range_min", d.frontend.range_min);
  c.frontend.range_max = real("range_max", d.frontend.range_max);
  c.frontend.rcs_iqr_k = real("rcs_iqr_k", d.frontend.rcs_iqr_k);
  c.frontend.lambda_z = real("lambda_z", d.frontend.lambda_z);
  c.frontend.huber_delta = real("huber_delta", d.frontend.huber_delta);
  c.frontend.irls_iterations = integer("irls_iterations", d.frontend.irls_iterations);

  c.matcher.mode = static_cast<MatcherMode>(
      choice("matcher", {"none", "scan", "submap"}, static_cast<size_t>(d.matcher.mode)));
  c.matcher.dof = static_cast<MatcherDof>(
      choice("matcher_dof", {"yaw", "planar"}, static_cast<size_t>(d.matcher.dof)));
  c.matcher.submap_size = integer("submap_size", d.matcher.submap_size);
  c.matcher.keyframe_distance = real("keyframe_distance", d.matcher.keyframe_distance);
  c.matcher.keyframe_angle = degrees("keyframe_angle_deg", d.matcher.keyframe_angle);
  c.matcher.sigma_range = real("sigma_range", d.matcher.sigma_range);
  c.matcher.sigma_azimuth = degrees("sigma_azimuth_deg", d.matcher.sigma_azimuth);
  c.matcher.max_correspondence_distance =
      real("max_correspondence_distance", d.matcher.max_correspondence_distance);
  c.matcher.robust_kernel = real("robust_kernel", d.matcher.robust_kernel);
  c.matcher.prior_yaw_std = degrees("prior_yaw_std_deg", d.matcher.prior_yaw_std);
  c.matcher.prior_xy_std = real("prior_xy_std", d.matcher.prior_xy_std);
  c.matcher.max_iterations = integer("matcher_max_iterations", d.matcher.max_iterations);
  c.matcher.min_correspondences = integer("min_correspondences", d.matcher.min_correspondences);
  c.matcher.max_yaw_correction =
      degrees("max_yaw_correction_deg", d.matcher.max_yaw_correction);

  c.scan_yaw_std = degrees("scan_yaw_std_deg", d.scan_yaw_std);
  c.scan_yaw_gate = real("scan_yaw_gate", d.scan_yaw_gate);

  const auto vel_std = this->declare_parameter<std::vector<double>>(
      full("radar_vel_std"),
      {d.rio.radar_vel_std.x(), d.rio.radar_vel_std.y(), d.rio.radar_vel_std.z()});
  if (vel_std.size() != 3) {
    throw std::invalid_argument(prefix + "radar_vel_std must have 3 elements");
  }
  c.rio.radar_vel_std = Eigen::Vector3d(vel_std[0], vel_std[1], vel_std[2]);
  c.rio.nis_gate = real("nis_gate", d.rio.nis_gate);
  c.rio.max_consecutive_rejections =
      integer("max_consecutive_rejections", d.rio.max_consecutive_rejections);

  if (c.frontend.inlier_thresh_mps <= 0.0 || c.frontend.gnc_mu_step <= 1.0 ||
      c.matcher.sigma_range <= 0.0 || c.matcher.prior_yaw_std <= 0.0 ||
      c.matcher.prior_xy_std <= 0.0 || c.matcher.robust_kernel <= 0.0 ||
      c.matcher.submap_size < 1 || c.scan_yaw_std <= 0.0) {
    throw std::invalid_argument("run '" + prefix + "' has an out-of-range setting");
  }
  return c;
}

void RadomNode::checkOverrides() {
  const auto &all = this->get_node_parameters_interface()->get_parameter_overrides();
  for (const auto &pipeline : pipelines_) {
    const std::string prefix = pipeline.name + ".";
    for (const auto &entry : all) {
      if (entry.first.rfind(prefix, 0) == 0 && known_parameters_.count(entry.first) == 0) {
        throw std::invalid_argument("unknown setting '" + entry.first + "'");
      }
    }
  }
}

void RadomNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  const rclcpp::Time stamp(msg->header.stamp);

  ImuMeasurement imu;
  imu.stamp_ns = stamp.nanoseconds();
  imu.timestamp = stamp.seconds();
  imu.linear_accel = Eigen::Vector3d(
      msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  imu.angular_vel = Eigen::Vector3d(
      msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
  imu.orientation = Eigen::Quaterniond(
      msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);

  std::lock_guard<std::mutex> lock(mutex_);
  if (last_imu_stamp_ns_ && imu.stamp_ns <= *last_imu_stamp_ns_) {
    if (*last_imu_stamp_ns_ - imu.stamp_ns < kTimeJumpNs) return;  // out of order
    RCLCPP_WARN(this->get_logger(), "IMU time jumped backwards, clearing buffers");
    imu_buffer_.clear();
    pending_scans_.clear();
    prev_imu_.reset();
  }
  last_imu_stamp_ns_ = imu.stamp_ns;

  if (imu_buffer_.size() >= 5000) {
    imu_buffer_.pop_front();
  }
  imu_buffer_.push_back(std::move(imu));
  processPending();
}

void RadomNode::radarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  const auto *fx = findField(*msg, "x");
  const auto *fy = findField(*msg, "y");
  const auto *fz = findField(*msg, "z");
  const auto *fd = findField(*msg, doppler_field_);
  const auto *fr = findField(*msg, rcs_field_);
  const auto *fi = findField(*msg, kInjectedField);

  const size_t num_points = static_cast<size_t>(msg->width) * msg->height;

  // Without Doppler every point would read as zero velocity, so drop the scan
  if (!readable(*msg, fx) || !readable(*msg, fy) || !readable(*msg, fz) ||
      !readable(*msg, fd) || msg->is_bigendian ||
      msg->row_step < static_cast<size_t>(msg->width) * msg->point_step ||
      msg->data.size() < static_cast<size_t>(msg->row_step) * msg->height) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "dropping radar scan: needs x, y, z and '%s' fields in a well-formed cloud",
        doppler_field_.c_str());
    return;
  }
  const bool has_rcs = readable(*msg, fr);
  const bool has_label = readable(*msg, fi);
  if (!has_rcs) {
    RCLCPP_WARN_ONCE(this->get_logger(), "no usable '%s' field, RCS filtering is off",
                     rcs_field_.c_str());
  }

  // Shift the radar stamp onto the IMU clock
  const rclcpp::Time stamp(msg->header.stamp);
  RadarScan scan;
  scan.stamp_ns = stamp.nanoseconds() + static_cast<int64_t>(radar_time_offset_s_ * 1e9);
  if (scan.stamp_ns < 0) return;
  scan.timestamp = static_cast<double>(scan.stamp_ns) * 1e-9;
  scan.points.reserve(num_points);

  for (size_t row = 0; row < msg->height; ++row) {
    for (size_t col = 0; col < msg->width; ++col) {
      const uint8_t *data = msg->data.data() + row * msg->row_step + col * msg->point_step;
      RadarPoint pt;
      pt.x = readValue(data, *fx);
      pt.y = readValue(data, *fy);
      pt.z = readValue(data, *fz);
      pt.doppler = static_cast<float>(readValue(data, *fd));
      if (has_rcs) pt.rcs = static_cast<float>(readValue(data, *fr));
      if (has_label) pt.injected = readValue(data, *fi) != 0.0;
      scan.points.push_back(pt);
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (last_radar_stamp_ns_ && *last_radar_stamp_ns_ - scan.stamp_ns > kTimeJumpNs) {
    pending_scans_.clear();
  }
  last_radar_stamp_ns_ = scan.stamp_ns;

  if (pending_scans_.size() >= kMaxPendingScans) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "no IMU data covering the radar scans, dropping the oldest");
    pending_scans_.pop_front();
  }
  pending_scans_.push_back(std::move(scan));
  processPending();
}

std::vector<ImuMeasurement> RadomNode::takeImuUpTo(int64_t stamp_ns) {
  std::vector<ImuMeasurement> batch;
  while (!imu_buffer_.empty() && imu_buffer_.front().stamp_ns <= stamp_ns) {
    prev_imu_ = imu_buffer_.front();
    batch.push_back(std::move(imu_buffer_.front()));
    imu_buffer_.pop_front();
  }

  // Interpolate a sample exactly at the scan time so the filter state and the
  // lever-arm gyro reading both refer to the instant of the radar measurement
  if (prev_imu_ && prev_imu_->stamp_ns < stamp_ns && !imu_buffer_.empty()) {
    const ImuMeasurement &next = imu_buffer_.front();
    const double a = static_cast<double>(stamp_ns - prev_imu_->stamp_ns) /
                     static_cast<double>(next.stamp_ns - prev_imu_->stamp_ns);
    ImuMeasurement mid = *prev_imu_;
    mid.stamp_ns = stamp_ns;
    mid.timestamp = static_cast<double>(stamp_ns) * 1e-9;
    mid.angular_vel += a * (next.angular_vel - prev_imu_->angular_vel);
    mid.linear_accel += a * (next.linear_accel - prev_imu_->linear_accel);
    mid.orientation = prev_imu_->orientation.slerp(a, next.orientation);
    batch.push_back(mid);
    prev_imu_ = mid;
  }
  return batch;
}

void RadomNode::processPending() {
  while (!pending_scans_.empty() && !imu_buffer_.empty() &&
         imu_buffer_.back().stamp_ns >= pending_scans_.front().stamp_ns) {
    const RadarScan scan = std::move(pending_scans_.front());
    pending_scans_.pop_front();
    const auto imu_batch = takeImuUpTo(scan.stamp_ns);

    // Rotate which run goes first, so that warm caches do not consistently
    // favour one of them in the runtime comparison
    const size_t count = pipelines_.size();
    for (size_t k = 0; k < count; ++k) {
      const size_t i = (scan_count_ + k) % count;
      // Only one estimator can own the TF
      runPipeline(pipelines_[i], scan, imu_batch, pipelines_[i].name == tf_run_);
    }
    ++scan_count_;
  }
}

void RadomNode::runPipeline(Pipeline &pipeline, const RadarScan &scan,
                            const std::vector<ImuMeasurement> &imu_batch, bool send_tf) {
  try {
    const auto result = pipeline.estimator->process(scan, imu_batch);
    if (!result) return;
    const UpdateResult update = pipeline.estimator->lastStats().velocity_update;
    if (update == UpdateResult::kRejected) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "%s: radar velocity rejected by the innovation gate",
                           pipeline.name.c_str());
    } else if (update == UpdateResult::kRecovering) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "%s: repeated rejections, inflating velocity covariance",
                            pipeline.name.c_str());
    }
    publishOdometry(pipeline, *result, send_tf);
    publishStats(pipeline, scan.stamp_ns);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(), "%s: estimator failed, resetting: %s",
                 pipeline.name.c_str(), e.what());
    pipeline.estimator = std::make_unique<OdometryEstimator>(pipeline.config);
  }
}

void RadomNode::publishStats(const Pipeline &pipeline, int64_t stamp_ns) {
  const EstimatorStats &stats = pipeline.estimator->lastStats();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = pipeline.name;
  const auto add = [&status](const std::string &key, const std::string &value) {
    diagnostic_msgs::msg::KeyValue kv;
    kv.key = key;
    kv.value = value;
    status.values.push_back(kv);
  };
  add("points", std::to_string(stats.frontend.num_points));
  add("inliers", std::to_string(stats.frontend.num_inliers));
  add("iterations", std::to_string(stats.frontend.iterations));
  add("runtime_us", std::to_string(stats.frontend.runtime_us));
  add("accepted", stats.velocity_update == UpdateResult::kAccepted ? "1" : "0");
  add("injected_points", std::to_string(stats.frontend.injected_points));
  add("injected_inliers", std::to_string(stats.frontend.injected_inliers));
  if (stats.velocity.valid) {
    add("vx", std::to_string(stats.velocity.velocity.x()));
    add("vy", std::to_string(stats.velocity.velocity.y()));
    add("vz", std::to_string(stats.velocity.velocity.z()));
  }
  if (pipeline.config.matcher.mode != MatcherMode::kNone) {
    add("match_runtime_us", std::to_string(stats.match_runtime_us));
    add("matched", stats.match.valid ? "1" : "0");
    add("heading_used", stats.heading_used ? "1" : "0");
    add("correspondences", std::to_string(stats.match.correspondences));
    add("match_yaw_deg", std::to_string(stats.match.yaw / kDegToRad));
  }

  diagnostic_msgs::msg::DiagnosticArray msg;
  msg.header.stamp = rclcpp::Time(stamp_ns, RCL_ROS_TIME);
  msg.status.push_back(status);
  pipeline.stats_pub->publish(msg);
}

void RadomNode::publishOdometry(const Pipeline &pipeline, const OdometryState &state,
                                bool send_tf) {
  const rclcpp::Time stamp(state.stamp_ns, RCL_ROS_TIME);
  const Eigen::Quaterniond q = state.orientation.normalized();

  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = stamp;
  odom_msg.header.frame_id = odom_frame_;
  odom_msg.child_frame_id = base_frame_;

  odom_msg.pose.pose.position.x = state.position.x();
  odom_msg.pose.pose.position.y = state.position.y();
  odom_msg.pose.pose.position.z = state.position.z();
  odom_msg.pose.pose.orientation.w = q.w();
  odom_msg.pose.pose.orientation.x = q.x();
  odom_msg.pose.pose.orientation.y = q.y();
  odom_msg.pose.pose.orientation.z = q.z();

  odom_msg.twist.twist.linear.x = state.linear_velocity.x();
  odom_msg.twist.twist.linear.y = state.linear_velocity.y();
  odom_msg.twist.twist.linear.z = state.linear_velocity.z();
  odom_msg.twist.twist.angular.x = state.angular_velocity.x();
  odom_msg.twist.twist.angular.y = state.angular_velocity.y();
  odom_msg.twist.twist.angular.z = state.angular_velocity.z();

  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      odom_msg.pose.covariance[r * 6 + c] = state.pose_covariance(r, c);
      odom_msg.twist.covariance[r * 6 + c] = state.twist_covariance(r, c);
    }
  }

  pipeline.odom_pub->publish(odom_msg);

  if (send_tf && publish_tf_ && tf_broadcaster_) {
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = odom_frame_;
    tf_msg.child_frame_id = base_frame_;
    tf_msg.transform.translation.x = state.position.x();
    tf_msg.transform.translation.y = state.position.y();
    tf_msg.transform.translation.z = state.position.z();
    tf_msg.transform.rotation = odom_msg.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf_msg);
  }
}

}