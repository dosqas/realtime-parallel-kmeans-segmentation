#pragma once
#include <opencv2/core.hpp>

// Create a 5D feature vector from BGR color and normalized spatial coordinates, scaled by the color_scale and
// spatial_scale arguments respectively
//
// Args:
//   bgr: the BGR color of the pixel (Vec3f)
//   x01, y01: normalized spatial coordinates of the pixel in the frame ([0, 1] range)
//   color_scale: scaling factor for the color dimensions
//   spatial_scale: scaling factor for the spatial dimensions
//
// Returns:
//   A 5D feature vector (Vec<float, 5>) with scaled color and spatial components
cv::Vec<float, 5> makeFeature(
	const cv::Vec3f& bgr, 
	float x01, 
	float y01, 
	float color_scale, 
	float spatial_scale
);

// Compute the K-means centers of a frame, by building and using a coreset of at most sample_size points
//
// Args:
//   frame: input image (cv::Mat, 3-channel BGR)
//   k: number of clusters for K-means
//   sample_size: maximum number of points in the coreset
//   color_scale: scaling factor for the color dimensions in the feature vectors
//   spatial_scale: scaling factor for the spatial dimensions in the feature vectors
std::vector<cv::Vec<float, 5>> computeKMeansCenters(
	const cv::Mat& frame,
	int k,
	int sample_size,
	float color_scale = 1.0f,
	float spatial_scale = 0.5f
);