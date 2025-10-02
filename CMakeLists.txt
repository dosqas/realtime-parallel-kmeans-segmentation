cmake_minimum_required(VERSION 3.10)
project(RealtimeParallelKMeansSegmentation)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(video_io src/main.cpp src/video_io.cpp "src/coreset.cpp")
target_include_directories(video_io PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(video_io ${OpenCV_LIBS})