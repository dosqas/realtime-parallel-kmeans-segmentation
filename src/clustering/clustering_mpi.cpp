#include "clustering_backends.hpp"
#include "coreset.hpp"
#include "utils.hpp"
#include <random>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <opencv2/opencv.hpp>

// Segment a frame using K-means clustering (distributed implementation)
// Uses Message Passing Interface (MPI) for distributed processing across multiple processes/nodes
cv::Mat segmentFrameWithKMeans_mpi(
    const cv::Mat& frame,
    int k,
    int sample_size,
    float color_scale,
    float spatial_scale)
{
    CV_Assert(frame.type() == CV_8UC3); // Ensure 3-channel BGR image

	// Ranks represent different processes in MPI
    int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
	MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    int rows = frame.rows;
    int cols = frame.cols;

    // 1. Compute centers on rank 0
    std::vector<float> flatCenters(k * 5, 0.0f); // Flattened 5D centers
    if (rank == 0) {
        auto centers = computeKMeansCenters(frame, k, sample_size, color_scale, spatial_scale);
        for (int i = 0; i < k; ++i)
            for (int d = 0; d < 5; ++d)
                flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(i) * 5 + d] = centers[i][d];
    }

    // 2. Broadcast centers to all ranks
	// Ensure all ranks wait until rank 0 is done computing centers
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(flatCenters.data(), k * 5, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 3. Determine rows for this rank
    int rowsPerRank = rows / size;
    int startRow = rank * rowsPerRank;
    int endRow = (rank == size - 1) ? rows : startRow + rowsPerRank;

    // 4. Flatten input chunk
    std::vector<uchar> localFlat((endRow - startRow) * cols * 3);
#pragma omp parallel for schedule(dynamic)
    for (int r = startRow; r < endRow; ++r) {
        const cv::Vec3b* inRow = frame.ptr<cv::Vec3b>(r);
        for (int c = 0; c < cols; ++c) {
            int idx = ((r - startRow) * cols + c) * 3;
            localFlat[static_cast<std::vector<uchar, std::allocator<uchar>>::size_type>(idx) + 0] = inRow[c][0];
            localFlat[static_cast<std::vector<uchar, std::allocator<uchar>>::size_type>(idx) + 1] = inRow[c][1];
            localFlat[static_cast<std::vector<uchar, std::allocator<uchar>>::size_type>(idx) + 2] = inRow[c][2];
        }
    }

    // 5. Process local chunk
    std::vector<uchar> localResult(localFlat.size());
#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < (endRow - startRow); ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = (r * cols + c) * 3;
            float f[5] = {
                localFlat[idx + 0] * color_scale,
                localFlat[idx + 1] * color_scale,
                localFlat[idx + 2] * color_scale,
                float(c) / float(cols) * spatial_scale,
                float(r + startRow) / float(rows) * spatial_scale
            };

            int bestIdx = 0;
            float bestDist2 = std::numeric_limits<float>::max();
            for (int ci = 0; ci < k; ++ci) {
                float d2 = 0.f;
                for (int d = 0; d < 5; ++d) {
                    float diff = f[d] - flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(ci) * 5 + d];
                    d2 += diff * diff;
                }
                if (d2 < bestDist2) { bestDist2 = d2; bestIdx = ci; }
            }

            localResult[static_cast<std::vector<uchar, std::allocator<uchar>>::size_type>(idx) + 0] = static_cast<uchar>(
                std::min(255.f, flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(bestIdx) * 5 + 0] / std::max(1e-6f, color_scale)));
            localResult[static_cast<std::vector<uchar, std::allocator<uchar>>::size_type>(idx) + 1] = static_cast<uchar>(
                std::min(255.f, flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(bestIdx) * 5 + 1] / std::max(1e-6f, color_scale)));
            localResult[static_cast<std::vector<uchar, std::allocator<uchar>>::size_type>(idx) + 2] = static_cast<uchar>(
                std::min(255.f, flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(bestIdx) * 5 + 2] / std::max(1e-6f, color_scale)));
        }
    }

    // 6. Prepare counts and displacements for MPI_Gatherv
	// counts: how many elements each rank will send
	// displs: where in the final array each rank's data will go
	// The data we send is 3 channels (BGR) per pixel
    std::vector<int> recvCounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        int sRow = i * rowsPerRank;
        int eRow = (i == size - 1) ? rows : sRow + rowsPerRank;
        recvCounts[i] = (eRow - sRow) * cols * 3;
        displs[i] = sRow * cols * 3;
    }

    // 7. Gather all results at rank 0
    cv::Mat out;
    if (rank == 0)
        out.create(rows, cols, frame.type());

	// Gatherv is like a backwards BCast - each rank sends its data to rank 0, which collects it all
	// v comes from variable-sized - each rank may have a different amount of data to send
    MPI_Gatherv(localResult.data(), localResult.size(), MPI_UNSIGNED_CHAR,
        (rank == 0 ? out.data : nullptr), recvCounts.data(), displs.data(),
        MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    return out; // Only rank 0 will have the full image filled
}
