# CMake is a cross-platform build system generator.
# This CMakeLists.txt file sets up the build configuration for the Realtime Parallel K-Means Segmentation project.
# It specifies the minimum required CMake version, project name, source files, and dependencies.
# The most recent version of CMake can be downloaded from https://cmake.org/download/
cmake_minimum_required(VERSION 3.18)
project(RealtimeParallelKMeansSegmentation LANGUAGES CXX CUDA)

# The OpenCV library is required for this project, for facilitating the real-time image processing and segmentation tasks.
find_package(OpenCV REQUIRED)

find_package(CUDA REQUIRED)

# Find Microsoft MPI
set(MPI_CXX_INCLUDE_PATH "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MPI_CXX_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")
# Include OpenCV and MPI headers, to allow access to both OpenCV and MS MPI functions and classes.
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${MPI_CXX_INCLUDE_PATH})

enable_language(CUDA)

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
    gpu/clustering_cuda.cu
)

set_source_files_properties(gpu/clustering_cuda.cu PROPERTIES LANGUAGE CUDA)


# Link the OpenCV libraries to the executable target, ensuring that the necessary OpenCV functionality is available at runtime.
target_include_directories(realtime_parallel_kmeans_segmentation
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDA_INCLUDE_DIRS}
)
set_target_properties(realtime_parallel_kmeans_segmentation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)

set_target_properties(realtime_parallel_kmeans_segmentation PROPERTIES CUDA_ARCHITECTURES "native")
target_link_libraries(realtime_parallel_kmeans_segmentation ${OpenCV_LIBS} ${MPI_CXX_LIBRARIES})