cmake_minimum_required(VERSION 3.13)
project(Camera_calibration)


find_package(OpenCV 3.3 REQUIRED)
find_package(Boost      REQUIRED COMPONENTS system chrono filesystem unit_test_framework program_options)

include_directories(${Boost_INCLUDE_DIR})
link_directories   (${Boost_LIBRARY_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})





set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Camera_calibration main.cpp)

target_link_libraries(Camera_calibration ${OpenCV_LIBS} ${Boost_LIBRARIES})