#pragma once
#include <opencv2/core.hpp>

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