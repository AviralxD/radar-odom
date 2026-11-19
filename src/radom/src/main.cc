#include <memory>
#include <rclcpp/rclcpp.hpp>
#include "radom/radom_node.hh"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<radom::RadomNode>());
  rclcpp::shutdown();
  return 0;
}