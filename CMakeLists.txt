cmake_minimum_required(VERSION 3.0.2)
project(mpc_model_id_mismatch)

find_package(catkin REQUIRED COMPONENTS
  roscpp
)

catkin_python_setup()

catkin_package(
  # CATKIN_DEPENDS roscpp
  # INCLUDE_DIRS include/${PROJECT_NAME}
  LIBRARIES ${PROJECT_NAME}
)

include_directories(
  include/${PROJECT_NAME}
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/rosbag_to_json.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
