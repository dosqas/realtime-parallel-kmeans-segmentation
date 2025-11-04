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

// Compute the K-means centers of a frame, by building and using a coreset of at most sample_size points
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

// Worker function used to process a range of rows in the image (used in both multithreaded implementations)
void processRows(
	const cv::Mat& frame,
	cv::Mat& out,
	const std::vector<cv::Vec<float, 5>>& centers,
	int rStart,
	int rEnd,
	float color_scale,
	float spatial_scale)
{
	int cols = frame.cols;
	int rows = frame.rows;

	// First go row by row
	for (int r = rStart; r < rEnd; ++r) {
		const cv::Vec3b* inRow = frame.ptr<cv::Vec3b>(r); // Pointer to the current row in input image
		cv::Vec3b* outRow = out.ptr<cv::Vec3b>(r); // Pointer to the current row in output image
		float y01 = (float)r / (float)rows; // Normalized y coordinate

		// Then go pixel by pixel in the row
		for (int c = 0; c < cols; ++c) {
			const cv::Vec3b& pix = inRow[c]; // Current pixel color
			float x01 = (float)c / (float)cols; // Normalized x coordinate

			// Create a 5D feature vector for the pixel
			cv::Vec<float, 5> f = makeFeature(
				cv::Vec3f(pix[0], pix[1], pix[2]),
				x01, y01,
				color_scale, spatial_scale);

			int bestIdx = 0;
			float bestDist2 = std::numeric_limits<float>::max();
			// Find the nearest K-means center to the pixel's feature vector by going through all centers
			for (int ci = 0; ci < (int)centers.size(); ++ci) {
				float d2 = 0.0f;
				// Compute squared Euclidean distance in 5D space for each of the 5 dimensions (BGRXY)
				for (int d = 0; d < 5; ++d) {
					float diff = f[d] - centers[ci][d];
					d2 += diff * diff;
				}
				if (d2 < bestDist2) { bestDist2 = d2; bestIdx = ci; }
			}

			cv::Vec3b color;
			// Determine the output pixel color as the color of the nearest center, scaled back by the color_scale factor
			color[0] = (uchar)cv::saturate_cast<uchar>(centers[bestIdx][0] / std::max(1e-6f, color_scale));
			color[1] = (uchar)cv::saturate_cast<uchar>(centers[bestIdx][1] / std::max(1e-6f, color_scale));
			color[2] = (uchar)cv::saturate_cast<uchar>(centers[bestIdx][2] / std::max(1e-6f, color_scale));
			outRow[c] = color;
		}
	}
}