#pragma once
#include <opencv2/core.hpp>

// Create a 5D feature vector from BGR color and normalized spatial coordinates, scaled by the color_scale and
// spatial_scale arguments respectively
cv::Vec<float, 5> makeFeature(
	const cv::Vec3f& bgr, 
	float x01, 
	float y01, 
	float color_scale, 
	float spatial_scale
);

// Compute the K-means centers of a frame, by building and using a coreset of at most sample_size points
std::vector<cv::Vec<float, 5>> computeKMeansCenters(
	const cv::Mat& frame,
	int k,
	int sample_size,
	float color_scale = 1.0f,
	float spatial_scale = 0.5f
);