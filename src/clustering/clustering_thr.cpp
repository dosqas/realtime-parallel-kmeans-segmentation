#include "clustering_backends.hpp"
#include "coreset.hpp"
#include "utils.hpp"
#include <random>

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
