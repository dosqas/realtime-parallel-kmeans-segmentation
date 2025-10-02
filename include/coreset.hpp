#pragma once
#include <opencv2/core.hpp>

struct CoresetPoint {
	cv::Vec3f rgb;
	float x, y;
	float weight;
};

struct Coreset {
	std::vector<CoresetPoint> points;
};

Coreset buildCoresetFromFrame(const cv::Mat& frame, int sample_size);

Coreset mergeCoresets(const Coreset& A, const Coreset& B, int sample_size);