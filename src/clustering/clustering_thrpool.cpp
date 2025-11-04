#include "clustering_backends.hpp"
#include "coreset.hpp"
#include "utils.hpp"
#include "threadpool.hpp"

// Segment a frame using K-means clustering (threaded implementation, with pooling)
cv::Mat segmentFrameWithKMeans_thrpool(
    const cv::Mat& frame,
    int k,
    int sample_size,
    float color_scale,
    float spatial_scale)
{
    CV_Assert(frame.type() == CV_8UC3); // Ensure 3-channel BGR image

    // Compute K-means centers using a coreset
    std::vector<cv::Vec<float, 5>> centers =
        computeKMeansCenters(frame, k, sample_size, color_scale, spatial_scale);

    cv::Mat out(frame.size(), frame.type());
    int rows = frame.rows;

    // Decide number of threads for the pool
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    int chunkSize = rows / numThreads; // Rows per thread

    // Enqueue tasks for each chunk of rows
    for (unsigned int t = 0; t < numThreads; ++t) {
        int rStart = t * chunkSize;
        int rEnd = (t == numThreads - 1) ? rows : rStart + chunkSize; // Last thread may take extra rows

        getThreadPool().enqueue([&, rStart, rEnd]() {
            processRows(frame, out, centers, rStart, rEnd, color_scale, spatial_scale);
            });
    }

    // Wait until all tasks are finished
    getThreadPool().waitUntilEmpty();

    return out;
}
