#include "clustering.hpp"
#include "clustering_backends.hpp"

// Entry point to segment a frame using K-means clustering with different backends
cv::Mat segmentFrameWithKMeans(
    const cv::Mat& frame,
    int k,
    Backend backend,
    int sample_size,
    float color_scale,
    float spatial_scale,
    const std::vector<float>& flatCenters,
    int totalRows,
    int totalCols)
{
	// Dispatch to the appropriate backend implementation
    switch (backend) {
    case BACKEND_SEQ:
        return segmentFrameWithKMeans_seq(frame, k, sample_size, color_scale, spatial_scale);
    case BACKEND_THR:
        return segmentFrameWithKMeans_thr(frame, k, sample_size, color_scale, spatial_scale);
    case BACKEND_MPI:
        return segmentFrameWithKMeans_mpi(frame, k, color_scale, spatial_scale, flatCenters, totalRows, totalCols);
    case BACKEND_CUDA:
        return segmentFrameWithKMeans_cuda(frame, k, sample_size, color_scale, spatial_scale);
    case BACKEND_THRPOOL:
        return segmentFrameWithKMeans_thrpool(frame, k, sample_size, color_scale, spatial_scale);
    default:
        throw std::invalid_argument("Unknown backend type in segmentFrameWithKMeans");
    }
}
