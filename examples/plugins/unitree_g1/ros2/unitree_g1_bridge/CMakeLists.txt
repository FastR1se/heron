cmake_minimum_required(VERSION 3.16)
project(unitree_g1_bridge LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(THIS_PACKAGE_DEPENDS
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  unitree_hg
)

find_package(ament_cmake REQUIRED)
foreach(dependency IN ITEMS ${THIS_PACKAGE_DEPENDS})
  find_package(${dependency} REQUIRED)
endforeach()

add_library(unitree_g1_bridge SHARED src/low_level_bridge.cpp)
target_compile_features(unitree_g1_bridge PUBLIC cxx_std_17)
ament_target_dependencies(unitree_g1_bridge PUBLIC ${THIS_PACKAGE_DEPENDS})

# The controller manager finds the class through this description (ADR-0033).
pluginlib_export_plugin_description_file(controller_interface unitree_g1_bridge.xml)

install(
  TARGETS unitree_g1_bridge
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_package()
