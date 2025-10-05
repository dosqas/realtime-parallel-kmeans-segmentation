# CMake is a cross-platform build system generator.
# This CMakeLists.txt file sets up the build configuration for the Realtime Parallel K-Means Segmentation project.
# It specifies the minimum required CMake version, project name, source files, and dependencies.
# The most recent version of CMake can be downloaded from https://cmake.org/download/
cmake_minimum_required(VERSION 3.18)    # Even though 3.18 is an older version, it is sufficient for our purposes
project(RealtimeParallelKMeansSegmentation LANGUAGES CXX CUDA)    # Project name and languages used

# Find dependencies (OpenCV and CUDA)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Default Microsoft MPI paths (if installed elsewhere, modify these paths accordingly!)
# Used for compiling and linking MPI programs on Windows
set(MPI_INCLUDE_DIR "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MPI_LIB "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")

enable_language(CUDA)

# Include directories for OpenCV, MPI, CUDA and CMake, to be able to find header files
# and use classes and methods defined in those libraries
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${MPI_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# Add executable and specify source files used to build it
add_executable(realtime_parallel_kmeans_segmentation
    src/main.cpp
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

# CUDA properties to ensure proper compilation (we must compile our .cu files with the CUDA compiler)
set_source_files_properties(src/clustering/clustering_cuda.cu PROPERTIES LANGUAGE CUDA)
set_target_properties(realtime_parallel_kmeans_segmentation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON       # Enable separable compilation for CUDA (different from our .cpp files)
    CUDA_STANDARD 14                    # Use C++14 standard for CUDA files
    CUDA_STANDARD_REQUIRED ON           # Enforce the use of the specified C++ standard
    CUDA_ARCHITECTURES "native"         # Specify the target architecture (native for the current machine)
)

# Link the OpenCV and MPI libraries with our executable to resolve external references
target_link_libraries(realtime_parallel_kmeans_segmentation
    ${OpenCV_LIBS}
    ${MPI_LIB}
)