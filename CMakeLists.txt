# CMake is a cross-platform build system generator.
# This CMakeLists.txt file sets up the build configuration for the Realtime Parallel K-Means Segmentation project.
# It specifies the minimum required CMake version, project name, source files, and dependencies.
# The most recent version of CMake can be downloaded from https://cmake.org/download/
cmake_minimum_required(VERSION 3.18)    # Even though 3.18 is an older version, it is sufficient for our purposes
project(RealtimeParallelKMeansSegmentation LANGUAGES CXX CUDA)    # Project name and languages used

include(FetchContent)

# Find dependencies (OpenCV and CUDA)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Download and make GoogleTest available
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()

# Default Microsoft MPI paths (if installed elsewhere, modify these paths accordingly!)
# Used for compiling and linking MPI programs on Windows
set(MPI_INCLUDE_DIR "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MPI_LIB "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")

enable_language(CUDA)

# Include directories for OpenCV, MPI, CUDA and our project headers
# This ensures all targets can find and use the headers correctly
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${MPI_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# ======================================================================
# Create a static library for the core logic (shared between main app and tests)
# ======================================================================
add_library(kmeans_core
    src/video_io.cpp
    src/coreset.cpp
    src/rcc.cpp
    src/utils.cpp
    src/clustering/clustering_entry.cpp
    src/clustering/clustering_seq.cpp
    src/clustering/clustering_mpi.cpp
    src/clustering/clustering_thr.cpp
    src/clustering/clustering_cuda.cu
)

# CUDA properties to ensure proper compilation (must compile .cu files with the CUDA compiler)
set_source_files_properties(src/clustering/clustering_cuda.cu PROPERTIES LANGUAGE CUDA)
set_target_properties(kmeans_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON       # Enable separable compilation for CUDA (different from .cpp files)
    CUDA_STANDARD 14                    # Use C++14 standard for CUDA files
    CUDA_STANDARD_REQUIRED ON           # Enforce the use of the specified C++ standard
    CUDA_ARCHITECTURES "native"         # Specify target architecture (native for the current machine)
)

# Link necessary libraries for OpenCV, MPI, etc.
target_link_libraries(kmeans_core
    ${OpenCV_LIBS}
    ${MPI_LIB}
)

# ======================================================================
# Main executable target for the application
# ======================================================================
add_executable(realtime_parallel_kmeans_segmentation
    src/main.cpp
    cpp.hint
)

# Set CUDA properties for the main executable
set_target_properties(realtime_parallel_kmeans_segmentation PROPERTIES CUDA_ARCHITECTURES "native")

# Link the executable with the shared library
target_link_libraries(realtime_parallel_kmeans_segmentation PRIVATE kmeans_core)

# ======================================================================
# GoogleTest integration
# ======================================================================

# Gather all test source files
file(GLOB TEST_SOURCES
    tests/test_*.cpp
)

# Define the test executable
add_executable(run_tests ${TEST_SOURCES})

# Link the test executable with GTest, core library, and dependencies
target_link_libraries(run_tests
    PRIVATE
    gtest
    gtest_main
    kmeans_core
)

# Include project headers for the tests
target_include_directories(run_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Set CUDA properties for the test executable
set_target_properties(run_tests PROPERTIES CUDA_ARCHITECTURES "native")

# Enable GoogleTest automatic test discovery
include(GoogleTest)
gtest_discover_tests(run_tests)
