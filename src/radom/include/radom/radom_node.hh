#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
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
  // One estimator per front-end; all are fed identical scans and IMU batches
  struct Pipeline {
    std::string name;
    FrontendConfig config;
    std::unique_ptr<OdometryEstimator> estimator;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr stats_pub;
  };

  void radarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
  void publishOdometry(const Pipeline &pipeline, const OdometryState &state, bool send_tf);
  void publishStats(const Pipeline &pipeline, int64_t stamp_ns);
  void runPipeline(Pipeline &pipeline, const RadarScan &scan,
                   const std::vector<ImuMeasurement> &imu_batch, bool send_tf);
  // Run every queued scan that the IMU stream has caught up with (mutex_ held)
  void processPending();
  std::vector<ImuMeasurement> takeImuUpTo(int64_t stamp_ns);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr radar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::string odom_frame_;
  std::string base_frame_;
  std::string doppler_field_;
  std::string rcs_field_;
  bool publish_tf_ = true;

  double radar_time_offset_s_ = 0.0;
  RioConfig rio_config_;

  std::mutex mutex_;
  std::deque<ImuMeasurement> imu_buffer_;
  std::deque<RadarScan> pending_scans_;
  std::optional<ImuMeasurement> prev_imu_;  // last sample handed to the estimator
  std::optional<int64_t> last_imu_stamp_ns_;
  std::optional<int64_t> last_radar_stamp_ns_;
  std::vector<Pipeline> pipelines_;
  size_t scan_count_ = 0;
};

}