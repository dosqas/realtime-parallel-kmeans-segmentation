#include "utils.hpp"
#include "coreset.hpp"

cv::Vec<float, 5> makeFeature(const cv::Vec3f& bgr, float x01, float y01, float color_scale, float spatial_scale)
{
	return cv::Vec<float, 5>(
		bgr[0] * color_scale, 
		bgr[1] * color_scale, 
		bgr[2] * color_scale, 
		x01 * spatial_scale, 
		y01 * spatial_scale
	);
}

std::vector<cv::Vec<float, 5>> computeKMeansCenters(
	const cv::Mat& frame,
	int k,
	int sample_size,
	float color_scale,
	float spatial_scale)
{
	CV_Assert(frame.type() == CV_8UC3);
	if (k <= 0) k = 1;
	if (sample_size <= 0) sample_size = 1024;

	Coreset coreset = buildCoresetFromFrame(frame, sample_size);

	cv::Mat samples(static_cast<int>(coreset.points.size()), 5, CV_32F);
	for (int i = 0; i < (int)coreset.points.size(); ++i) {
		const CoresetPoint& p = coreset.points[i];
		cv::Vec<float, 5> f = makeFeature(p.rgb, p.x, p.y, color_scale, spatial_scale);
		for (int d = 0; d < 5; ++d) samples.at<float>(i, d) = f[d];
	}

	cv::Mat labels;
	cv::Mat centers;
	int attempts = 1;
	cv::kmeans(samples, k, labels,
		cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 20, 1e-3),
		attempts, cv::KMEANS_PP_CENTERS, centers);

	std::vector<cv::Vec<float, 5>> result;
	result.reserve(k);
	for (int i = 0; i < centers.rows; ++i) {
		cv::Vec<float, 5> c;
		for (int d = 0; d < 5; ++d) c[d] = centers.at<float>(i, d);
		result.push_back(c);
	}
	return result;
}