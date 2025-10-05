#pragma once
#include <opencv2/core.hpp>

// Segment a frame using K-means clustering with different backends (sequential, threaded, MPI, CUDA)
// See: clustering_seq.cpp, clustering_thr.cpp, clustering_mpi.cpp, clustering_cuda.cu for implementations
//      and include/clustering_backends.hpp for declarations/argument meanings

cv::Mat segmentFrameWithKMeans_seq(
	const cv::Mat& frame,
	int k,
	int sample_size,
	float color_scale = 1.0f,
	float spatial_scale = 0.5f
);

cv::Mat segmentFrameWithKMeans_thr(
	const cv::Mat& frame,
	int k,
	int sample_size,
	float color_scale = 1.0f,
	float spatial_scale = 0.5f
);

cv::Mat segmentFrameWithKMeans_mpi(
	const cv::Mat& frame,
	int k,
	int sample_size,
	float color_scale = 1.0f,
	float spatial_scale = 0.5f
);

cv::Mat segmentFrameWithKMeans_cuda(
	const cv::Mat& frame,
	int k,
	int sample_size,
	float color_scale = 1.0f,
	float spatial_scale = 0.5f
);