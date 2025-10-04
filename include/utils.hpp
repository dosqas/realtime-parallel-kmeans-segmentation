#pragma once
#include <opencv2/core.hpp>

cv::Vec<float, 5> makeFeature(
	const cv::Vec3f& bgr, 
	float x01, 
	float y01, 
	float color_scale, 
	float spatial_scale
);

std::vector<cv::Vec<float, 5>> computeKMeansCenters(
	const cv::Mat& frame,
	int k,
	int sample_size,
	float color_scale = 1.0f,
	float spatial_scale = 0.5f
);