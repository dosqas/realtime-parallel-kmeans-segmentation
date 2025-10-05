#pragma once
#include <opencv2/core.hpp>

// Backend options for k-means segmentation
// Sequential (CPU), CUDA (GPU), Threaded (CPU multi-threading), MPI (distributed)
enum Backend { BACKEND_SEQ = 0, BACKEND_CUDA = 1, BACKEND_THR = 2, BACKEND_MPI = 3 };

// Segment a frame using k-means clustering with the specified backend
//
// Args:
//   frame: input image (cv::Mat, 3-channel BGR)
//   k: number of clusters for K-means
//   sample_size: maximum number of points in the coreset
//   backend: which backend to use (sequential, CUDA, threaded, MPI)
//   color_scale: scaling factor for the color dimensions in the feature vectors
//   spatial_scale: scaling factor for the spatial dimensions in the feature vectors
//
// Returns:
//   Segmented image (cv::Mat, 3-channel BGR) where each pixel is colored by its cluster center
cv::Mat segmentFrameWithKMeans(
	const cv::Mat& frame,
	int k,
	int sample_size,
	Backend backend,
	float color_scale = 1.0f,
	float spatial_scale = 0.5f
);