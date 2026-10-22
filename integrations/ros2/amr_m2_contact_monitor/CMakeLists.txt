cmake_minimum_required(VERSION 3.10.2)
project(amr_m2_contact_monitor)
find_package(ament_cmake REQUIRED)
find_package(ignition-gazebo6 REQUIRED)
find_package(ignition-plugin1 REQUIRED COMPONENTS register)

add_library(m2_contact_monitor SHARED src/contact_monitor.cc)
target_compile_features(m2_contact_monitor PRIVATE cxx_std_17)
target_link_libraries(m2_contact_monitor PRIVATE
  ignition-gazebo6::ignition-gazebo6 ignition-plugin1::ignition-plugin1)
install(TARGETS m2_contact_monitor LIBRARY DESTINATION lib)

if(BUILD_TESTING)
  enable_testing()
  add_executable(test_m2_contacts test/test_contacts.cc)
  target_compile_features(test_m2_contacts PRIVATE cxx_std_17)
  target_include_directories(test_m2_contacts PRIVATE src)
  add_test(NAME m2_physics_contact_samples COMMAND test_m2_contacts)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME m2_fortress_contact_calibration
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/live_contacts.py
            $<TARGET_FILE_DIR:m2_contact_monitor>)
  set_tests_properties(m2_fortress_contact_calibration PROPERTIES TIMEOUT 70)
endif()
ament_package()
