#include "radom/radom_node.hh"

#include <algorithm>
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
  const auto odom_topic = this->declare_parameter<std::string>("odom_topic", "/radom/odom");
  odom_frame_ = this->declare_parameter<std::string>("odom_frame", "odom");
  base_frame_ = this->declare_parameter<std::string>("base_frame", "continental");
  publish_tf_ = this->declare_parameter<bool>("publish_tf", true);
  doppler_field_ = this->declare_parameter<std::string>("doppler_field", "v");
  rcs_field_ = this->declare_parameter<std::string>("rcs_field", "RCS");

  radar_time_offset_s_ = this->declare_parameter<double>("radar_time_offset_s", 0.0);
  const auto vel_std = this->declare_parameter<std::vector<double>>(
      "radar_vel_std", {rio_config_.radar_vel_std.x(), rio_config_.radar_vel_std.y(),
                        rio_config_.radar_vel_std.z()});
  if (vel_std.size() != 3) {
    throw std::invalid_argument("radar_vel_std must have 3 elements");
  }
  rio_config_.radar_vel_std = Eigen::Vector3d(vel_std[0], vel_std[1], vel_std[2]);
  rio_config_.nis_gate = this->declare_parameter<double>("nis_gate", rio_config_.nis_gate);
  rio_config_.max_consecutive_rejections = static_cast<int>(this->declare_parameter<int>(
      "max_consecutive_rejections", rio_config_.max_consecutive_rejections));
  estimator_ = std::make_unique<OdometryEstimator>(rio_config_);

  auto sensor_qos = rclcpp::SensorDataQoS();

  radar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      radar_topic, sensor_qos,
      std::bind(&RadomNode::radarCallback, this, std::placeholders::_1));

  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, rclcpp::SensorDataQoS().keep_last(200),
      std::bind(&RadomNode::imuCallback, this, std::placeholders::_1));

  odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic, 50);

  if (publish_tf_) {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  }

  RCLCPP_INFO(this->get_logger(), "radom started: [%s] + [%s] -> [%s]",
              radar_topic.c_str(), imu_topic.c_str(), odom_topic.c_str());
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

    try {
      const auto result = estimator_->process(scan, imu_batch);
      if (!result) continue;
      if (estimator_->lastUpdate() == UpdateResult::kRejected) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "radar velocity rejected by the innovation gate");
      } else if (estimator_->lastUpdate() == UpdateResult::kRecovering) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                              "repeated rejections, inflating velocity covariance");
      }
      publishOdometry(*result);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "estimator failed, resetting: %s", e.what());
      estimator_ = std::make_unique<OdometryEstimator>(rio_config_);
    }
  }
}

void RadomNode::publishOdometry(const OdometryState &state) {
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

  odom_pub_->publish(odom_msg);

  if (publish_tf_ && tf_broadcaster_) {
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