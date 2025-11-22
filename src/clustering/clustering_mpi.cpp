#include "clustering_backends.hpp"
#include "coreset.hpp"
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
    const cv::Mat& localFrame,
    int k,
    float color_scale,
    float spatial_scale,
    const std::vector<float>& flatCenters,
    int totalRows,
    int totalCols)
{
    CV_Assert(localFrame.type() == CV_8UC3);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = localFrame.rows;
    int cols = localFrame.cols;

    // Calculate startRow for this rank
    int rowsPerRank = totalRows / size;
    int startRow = rank * rowsPerRank;

    // Flatten local frame
    std::vector<uchar> localFlat(rows * cols * 3);
#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < rows; ++r) {
        const cv::Vec3b* inRow = localFrame.ptr<cv::Vec3b>(r);
        for (int c = 0; c < cols; ++c) {
            int idx = (r * cols + c) * 3;
            localFlat[static_cast<std::vector<uchar, std::allocator<uchar>>::size_type>(idx) + 0] = inRow[c][0];
            localFlat[static_cast<std::vector<uchar, std::allocator<uchar>>::size_type>(idx) + 1] = inRow[c][1];
            localFlat[static_cast<std::vector<uchar, std::allocator<uchar>>::size_type>(idx) + 2] = inRow[c][2];
        }
    }

    // Apply clustering using received centers
    std::vector<uchar> localResult(localFlat.size());

#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = (r * cols + c) * 3;

            float f[5] = {
                localFlat[idx + 0] * color_scale,
                localFlat[idx + 1] * color_scale,
                localFlat[idx + 2] * color_scale,
                float(c) / float(totalCols) * spatial_scale,
                float(r + startRow) / float(totalRows) * spatial_scale
            };

            int bestIdx = 0;
            float bestDist2 = std::numeric_limits<float>::max();

            for (int ci = 0; ci < k; ++ci) {
                float d2 = 0.f;
                for (int d = 0; d < 5; ++d) {
                    float diff = f[d] - flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(ci) * 5 + d];
                    d2 += diff * diff;
                }
                if (d2 < bestDist2) {
                    bestDist2 = d2;
                    bestIdx = ci;
                }
            }

            localResult[static_cast<std::vector<uchar, std::allocator<uchar>>::size_type>(idx) + 0] =
                (uchar)std::min(255.f, flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(bestIdx) * 5 + 0] / color_scale);
            localResult[static_cast<std::vector<uchar, std::allocator<uchar>>::size_type>(idx) + 1] =
                (uchar)std::min(255.f, flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(bestIdx) * 5 + 1] / color_scale);
            localResult[static_cast<std::vector<uchar, std::allocator<uchar>>::size_type>(idx) + 2] =
                (uchar)std::min(255.f, flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(bestIdx) * 5 + 2] / color_scale);
        }
    }

    // Gather segmentation results back to rank 0
    std::vector<int> recvCounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        int sRow = i * rowsPerRank;
        int eRow = (i == size - 1 ? totalRows : sRow + rowsPerRank);
        recvCounts[i] = (eRow - sRow) * cols * 3;
        displs[i] = sRow * cols * 3;
    }

    cv::Mat out;
    if (rank == 0)
        out.create(totalRows, totalCols, localFrame.type());

    MPI_Gatherv(
        localResult.data(), localResult.size(), MPI_UNSIGNED_CHAR,
        rank == 0 ? out.data : nullptr,
        recvCounts.data(), displs.data(),
        MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    return out;
}
