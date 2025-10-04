# CMake is a cross-platform build system generator.
# This CMakeLists.txt file sets up the build configuration for the Realtime Parallel K-Means Segmentation project.
# It specifies the minimum required CMake version, project name, source files, and dependencies.
# The most recent version of CMake can be downloaded from https://cmake.org/download/
cmake_minimum_required(VERSION 3.10)
project(RealtimeParallelKMeansSegmentation)

# The OpenCV library is required for this project, for facilitating the real-time image processing and segmentation tasks.
find_package(OpenCV REQUIRED)
# Include OpenCV headers, to allow access to OpenCV functions and classes.
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable target for the project, specifying the source files to be compiled.
add_executable(realtime_parallel_kmeans_segmentation
    src/main.cpp
    src/video_io.cpp
    src/coreset.cpp
    src/rcc.cpp
    src/utils.cpp
    src/clustering.cpp
    seq/clustering_seq.cpp
    threads/clustering_thr.cpp
    mpi/clustering_mpi.cpp
    gpu/clustering_cuda.cpp
 "threads/clustering_thr.cpp" "src/clustering.cpp" "mpi/clustering_mpi.cpp" "gpu/clustering_cuda.cpp")

# Link the OpenCV libraries to the executable target, ensuring that the necessary OpenCV functionality is available at runtime.
target_include_directories(realtime_parallel_kmeans_segmentation PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(realtime_parallel_kmeans_segmentation ${OpenCV_LIBS})