#include "radom/radom_node.hh"

#include <algorithm>
#include <vector>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace radom {

namespace {
bool hasField(const sensor_msgs::msg::PointCloud2 &msg, const std::string &name) {
  return std::any_of(msg.fields.begin(), msg.fields.end(),
                     [&name](const auto &f) { return f.name == name; });
}
}

RadomNode::RadomNode() : Node("radom") {
  const auto radar_topic = this->declare_parameter<std::string>("radar_topic", "/continental/points");
  const auto imu_topic = this->declare_parameter<std::string>("imu_topic", "/imu/data_raw");
  const auto odom_topic = this->declare_parameter<std::string>("odom_topic", "/radom/odom");
  odom_frame_ = this->declare_parameter<std::string>("odom_frame", "odom");
  base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
  publish_tf_ = this->declare_parameter<bool>("publish_tf", true);
  doppler_field_ = this->declare_parameter<std::string>("doppler_field", "v");
  rcs_field_ = this->declare_parameter<std::string>("rcs_field", "rcs");

  auto sensor_qos = rclcpp::SensorDataQoS();

  radar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      radar_topic, sensor_qos,
      std::bind(&RadomNode::radarCallback, this, std::placeholders::_1));

  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, sensor_qos,
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

  std::lock_guard<std::mutex> lock(imu_mutex_);
  if (imu_buffer_.size() >= 5000) {
    imu_buffer_.pop_front();
  }
  imu_buffer_.push_back(std::move(imu));
}

void RadomNode::radarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  const rclcpp::Time stamp(msg->header.stamp);

  RadarScan scan;
  scan.stamp_ns = stamp.nanoseconds();
  scan.timestamp = stamp.seconds();

  const size_t num_points = static_cast<size_t>(msg->width) * msg->height;
  scan.points.reserve(num_points);

  const bool has_doppler = hasField(*msg, doppler_field_);
  const bool has_rcs = hasField(*msg, rcs_field_);

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

  std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> iter_doppler;
  std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> iter_rcs;

  if (has_doppler) {
    iter_doppler = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, doppler_field_);
  }
  if (has_rcs) {
    iter_rcs = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, rcs_field_);
  }

  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    RadarPoint pt;
    pt.x = static_cast<double>(*iter_x);
    pt.y = static_cast<double>(*iter_y);
    pt.z = static_cast<double>(*iter_z);
    if (iter_doppler) {
      pt.doppler = **iter_doppler;
      ++(*iter_doppler);
    }
    if (iter_rcs) {
      pt.rcs = **iter_rcs;
      ++(*iter_rcs);
    }
    scan.points.push_back(pt);
  }

  // Drain all IMU measurements up to this radar scan's timestamp
  std::vector<ImuMeasurement> imu_batch;
  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    while (!imu_buffer_.empty() && imu_buffer_.front().stamp_ns <= scan.stamp_ns) {
      imu_batch.push_back(std::move(imu_buffer_.front()));
      imu_buffer_.pop_front();
    }
  }

  // Run odometry estimator and publish the result
  if (auto result = estimator_.process(scan, imu_batch)) {
    publishOdometry(*result);
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