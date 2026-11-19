#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "radom/odom.hh"
#include "radom/types.hh"

namespace radom {

class RadomNode : public rclcpp::Node {
public:
  RadomNode();

private:
  void radarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
  void publishOdometry(const OdometryState &state);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr radar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::string odom_frame_;
  std::string base_frame_;
  std::string doppler_field_;
  std::string rcs_field_;
  bool publish_tf_ = true;

  std::mutex imu_mutex_;
  std::deque<ImuMeasurement> imu_buffer_;
  OdometryEstimator estimator_;
};

}