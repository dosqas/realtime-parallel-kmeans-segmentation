#include "clustering.hpp"
#include "clustering_backends.hpp"

cv::Mat segmentFrameWithKMeans(
    const cv::Mat& frame,
    int k,
    int sample_size,
    Backend backend,
    float color_scale,
    float spatial_scale)
{
    switch (backend) {
    case BACKEND_SEQ:
        return segmentFrameWithKMeans_seq(frame, k, sample_size, color_scale, spatial_scale);
    case BACKEND_THR:
        return segmentFrameWithKMeans_thr(frame, k, sample_size, color_scale, spatial_scale);
    case BACKEND_MPI:
        return segmentFrameWithKMeans_mpi(frame, k, sample_size, color_scale, spatial_scale);
    case BACKEND_CUDA:
        return segmentFrameWithKMeans_cuda(frame, k, sample_size, color_scale, spatial_scale);
    default:
        throw std::invalid_argument("Unknown backend type in segmentFrameWithKMeans");
    }
}
