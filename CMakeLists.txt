cmake_minimum_required(VERSION 3.10)
project(RealtimeParallelKMeansSegmentation)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(video_io src/video_io.cpp)
target_link_libraries(video_io ${OpenCV_LIBS})
