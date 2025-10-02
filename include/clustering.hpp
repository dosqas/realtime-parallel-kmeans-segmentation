#pragma once
#include <opencv2/core.hpp>
#include <vector>

std::vector<cv::Vec<float, 5>> computeKMeansCenters(
	const cv::Mat& frame,
	int k,
	int sample_size,
	float color_scale = 1.0f,
	float spatial_scale = 0.5f
);

cv::Mat segmentFrameWithKMeans(
	const cv::Mat& frame,
	int k,
	int sample_size,
	float color_scale = 1.0f,
	float spatial_scale = 0.5f
);


