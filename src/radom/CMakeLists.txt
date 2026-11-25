cmake_minimum_required(VERSION 3.8)
project(radom)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# The front-end runtime comparison is meaningless without optimisation
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(ament_cmake REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)

set(EXTERN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../extern")
set(RANSACLIB_INCLUDE_DIR "${EXTERN_DIR}/ransac-lib")
set(MANIF_INCLUDE_DIR "${EXTERN_DIR}/manif/include")
set(MANIF_OPTIONAL_INCLUDE_DIR "${EXTERN_DIR}/manif/external/optional/include")
set(KALMANIF_INCLUDE_DIR "${EXTERN_DIR}/kalmanif/include")

add_executable(radom_node
  src/main.cc
  src/radom_node.cc
  src/odom.cc
  src/frontend.cc
  src/scan_matcher.cc
  src/radar_only.cc
  src/ransac.cc
  src/gnc.cc
  src/rio_iekf.cc
)

target_compile_features(radom_node PUBLIC cxx_std_17)

target_include_directories(radom_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(radom_node SYSTEM PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${RANSACLIB_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${MANIF_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${MANIF_OPTIONAL_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${KALMANIF_INCLUDE_DIR}>
)

ament_target_dependencies(radom_node
  rclcpp
  diagnostic_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
)

install(TARGETS radom_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch rviz DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/gt_viz.py DESTINATION lib/${PROJECT_NAME})

ament_package()