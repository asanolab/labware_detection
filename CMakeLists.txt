cmake_minimum_required(VERSION 3.0.2)
project(labware_detection)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  xarm_msgs
  message_filters
  geometry_msgs
  visualization_msgs
  lite6_labauto
)

catkin_package()

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/marker_publish.py
  scripts/pointcloud_merge.py
  scripts/publish_obb.py
  scripts/obb_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
