#pragma once
#include <opencv2/core.hpp>

enum Backend { BACKEND_SEQ = 0, BACKEND_CUDA = 1, BACKEND_THR = 2, BACKEND_MPI = 3 };

cv::Mat segmentFrameWithKMeans(
	const cv::Mat& frame,
	int k,
	int sample_size,
	Backend backend,
	float color_scale = 1.0f,
	float spatial_scale = 0.5f
);
