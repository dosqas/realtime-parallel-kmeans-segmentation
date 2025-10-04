#include "clustering_backends.hpp"
#include "coreset.hpp"
#include "utils.hpp"
#include <random>

cv::Mat segmentFrameWithKMeans_cuda(
	const cv::Mat& frame,
	int k,
	int sample_size,
	float color_scale,
	float spatial_scale)
{
	CV_Assert(frame.type() == CV_8UC3);

	std::vector<cv::Vec<float, 5>> centers = computeKMeansCenters(frame, k, sample_size, color_scale, spatial_scale);

	cv::Mat out(frame.size(), frame.type());

	int rows = frame.rows;
	int cols = frame.cols;
	for (int r = 0; r < rows; ++r) {
		const cv::Vec3b* inRow = frame.ptr<cv::Vec3b>(r);
		cv::Vec3b* outRow = out.ptr<cv::Vec3b>(r);
		float y01 = (float)r / (float)rows;
		for (int c = 0; c < cols; ++c) {
			const cv::Vec3b& pix = inRow[c];
			float x01 = (float)c / (float)cols;
			cv::Vec<float, 5> f = makeFeature(cv::Vec3f(pix[0], pix[1], pix[2]), x01, y01, color_scale, spatial_scale);

			int bestIdx = 0;
			float bestDist2 = std::numeric_limits<float>::max();
			for (int ci = 0; ci < (int)centers.size(); ++ci) {
				float d2 = 0.0f;
				for (int d = 0; d < 5; ++d) {
					float diff = f[d] - centers[ci][d];
					d2 += diff * diff;
				}
				if (d2 < bestDist2) { bestDist2 = d2; bestIdx = ci; }
			}

			cv::Vec3b color;
			color[0] = (uchar)cv::saturate_cast<uchar>(centers[bestIdx][0] / std::max(1e-6f, color_scale));
			color[1] = (uchar)cv::saturate_cast<uchar>(centers[bestIdx][1] / std::max(1e-6f, color_scale));
			color[2] = (uchar)cv::saturate_cast<uchar>(centers[bestIdx][2] / std::max(1e-6f, color_scale));
			outRow[c] = color;
		}
	}

	return out;
}


