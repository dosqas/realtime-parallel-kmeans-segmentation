#include "utils.hpp"
#include "coreset.hpp"

// Create a 5D feature vector from BGR color and normalized spatial coordinates, scaled by the color_scale and
// spatial_scale arguments
// We scale so that they are in roughly the same range and to not let color or space dominate the distance metric
cv::Vec<float, 5> makeFeature(
	const cv::Vec3f& bgr, 
	float x01, 
	float y01, 
	float color_scale, 
	float spatial_scale) 
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
	CV_Assert(frame.type() == CV_8UC3); // We assert that the input frame is a 3-channel BGR image
	if (k <= 0) k = 1;
	if (sample_size <= 0) sample_size = 1024;

	Coreset coreset = buildCoresetFromFrame(frame, sample_size); // Build the coreset of the frame

	cv::Mat samples(static_cast<int>(coreset.points.size()), 5, CV_32F); // Prepare a matrix to hold the 5D feature vectors
	for (int i = 0; i < (int)coreset.points.size(); ++i) {
		const CoresetPoint& p = coreset.points[i];
		cv::Vec<float, 5> f = makeFeature(p.bgr, p.x, p.y, color_scale, spatial_scale); // Make the feature vector of the CoresetPoint
		for (int d = 0; d < 5; ++d) samples.at<float>(i, d) = f[d]; // Copy it to the samples matrix
	}

	cv::Mat labels;
	cv::Mat centers;
	int attempts = 1;

	// Run K-means on the coreset samples to find k cluster centers in 5D space
	// We use KMEANS_PP_CENTERS for better initial center selection
	// We use a combination of EPS (convergence threshold) and MAX_ITER (max iterations) for termination criteria
	// EPS is set to 1e-3 and MAX_ITER to 20
	// We only do 1 attempt since the coreset is already a good summary of the data
	// The result is stored in 'centers' matrix
	cv::kmeans(samples, k, labels,
		cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 20, 1e-3),
		attempts, cv::KMEANS_PP_CENTERS, centers);

	std::vector<cv::Vec<float, 5>> result;
	result.reserve(k); // Reserve space for k centers
	for (int i = 0; i < centers.rows; ++i) {
		cv::Vec<float, 5> c;
		for (int d = 0; d < 5; ++d) c[d] = centers.at<float>(i, d);
		result.push_back(c);
	}
	return result;
}