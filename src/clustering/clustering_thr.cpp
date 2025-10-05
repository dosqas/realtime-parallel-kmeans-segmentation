#include "clustering_backends.hpp"
#include "coreset.hpp"
#include "utils.hpp"
#include <random>

#include <thread>
#include <vector>

// Worker function used to process a range of rows in the image
//
// Args:
//   frame: input image (cv::Mat, 3-channel BGR)
//   out: output segmented image (cv::Mat, 3-channel BGR)
//   centers: K-means centers (vector of 5D feature vectors)
//   rStart, rEnd: range of rows [rStart, rEnd) to process
//   color_scale: scaling factor for the color dimensions in the feature vectors
//   spatial_scale: scaling factor for the spatial dimensions in the feature vectors
static void processRows(
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

// Segment a frame using K-means clustering (threaded implementation)
cv::Mat segmentFrameWithKMeans_thr(
    const cv::Mat& frame,
    int k,
    int sample_size,
    float color_scale,
    float spatial_scale)
{
    CV_Assert(frame.type() == CV_8UC3); // Ensure input is 3-channel BGR image

    // Compute K-means centers using a coreset of the frame
    std::vector<cv::Vec<float, 5>> centers =
        computeKMeansCenters(frame, k, sample_size, color_scale, spatial_scale);

    cv::Mat out(frame.size(), frame.type());
    int rows = frame.rows;

    // Decide number of threads
	unsigned int numThreads = std::thread::hardware_concurrency(); // Typically returns number of CPU cores
    if (numThreads == 0) numThreads = 4; // Fallback

	int chunkSize = rows / numThreads; // Rows per thread

    // Launch workers
    std::vector<std::thread> workers;
    for (unsigned int t = 0; t < numThreads; ++t) {
        int rStart = t * chunkSize;
		int rEnd = (t == numThreads - 1) ? rows : rStart + chunkSize; // Last thread may take extra rows

		// Start thread
		// processRows is the worker function defined above
		// Capture frame and centers by const reference, out by reference
		// Allows safe concurrent reads of 'frame' and 'centers' (threads share them read-only)
		// Allows safe concurrent writes to different rows of 'out'
		// Also pass row range and scaling factors as arguments
        workers.emplace_back(processRows,
            std::cref(frame), std::ref(out),
            std::cref(centers),
            rStart, rEnd,
            color_scale, spatial_scale);
    }

	// Wait for all threads to finish
    for (auto& th : workers) th.join();

    return out;
}
