#include "clustering_backends.hpp"
#include "coreset.hpp"
#include "utils.hpp"
#include <random>
#include <mpi.h>

cv::Mat segmentFrameWithKMeans_mpi(
    const cv::Mat& frame,
    int k,
    int sample_size,
    float color_scale,
    float spatial_scale)
{
    CV_Assert(frame.type() == CV_8UC3);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<float> flatCenters;
    if (rank == 0) {
        auto centers = computeKMeansCenters(frame, k, sample_size, color_scale, spatial_scale);
        flatCenters.resize(k * 5);
        for (int i = 0; i < k; ++i)
            for (int d = 0; d < 5; ++d)
                flatCenters[i * 5 + d] = centers[i][d];
    }

    // Broadcast all centers in one go
    if (rank != 0) flatCenters.resize(k * 5);
    MPI_Bcast(flatCenters.data(), k * 5, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int rows = frame.rows;
    int cols = frame.cols;

    // Divide work
    int rowsPerRank = rows / size;
    int startRow = rank * rowsPerRank;
    int endRow = (rank == size - 1) ? rows : startRow + rowsPerRank;

    cv::Mat localOut(endRow - startRow, cols, frame.type());

    // Each rank processes its chunk in parallel
#pragma omp parallel for schedule(dynamic)
    for (int r = startRow; r < endRow; ++r) {
        const cv::Vec3b* inRow = frame.ptr<cv::Vec3b>(r);
        cv::Vec3b* outRow = localOut.ptr<cv::Vec3b>(r - startRow);
        float y01 = (float)r / (float)rows;
        for (int c = 0; c < cols; ++c) {
            const cv::Vec3b& pix = inRow[c];
            float x01 = (float)c / (float)cols;

            float f[5] = {
                pix[0] * color_scale, // B
                pix[1] * color_scale, // G
                pix[2] * color_scale, // R
                x01 * spatial_scale,
                y01 * spatial_scale
            };

            int bestIdx = 0;
            float bestDist2 = std::numeric_limits<float>::max();

            for (int ci = 0; ci < k; ++ci) {
                float d2 = 0.f;
#pragma unroll
                for (int d = 0; d < 5; ++d) {
                    float diff = f[d] - flatCenters[ci * 5 + d];
                    d2 += diff * diff;
                }
                if (d2 < bestDist2) { bestDist2 = d2; bestIdx = ci; }
            }

            cv::Vec3b color;
            color[2] = (uchar)std::min(255.f, flatCenters[bestIdx * 5 + 2] / std::max(1e-6f, color_scale)); // R
            color[1] = (uchar)std::min(255.f, flatCenters[bestIdx * 5 + 1] / std::max(1e-6f, color_scale)); // G
            color[0] = (uchar)std::min(255.f, flatCenters[bestIdx * 5 + 0] / std::max(1e-6f, color_scale)); // B
            outRow[c] = color;
        }
    }

    // Gather results at rank 0
    cv::Mat out;
    if (rank == 0) {
        out.create(rows, cols, frame.type());
    }

    std::vector<int> recvCounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        int sRow = i * rowsPerRank;
        int eRow = (i == size - 1) ? rows : sRow + rowsPerRank;
        recvCounts[i] = (eRow - sRow) * cols * 3; // bytes
        displs[i] = sRow * cols * 3;
    }

    MPI_Gatherv(localOut.data,
        (endRow - startRow) * cols * 3, MPI_UNSIGNED_CHAR,
        out.data, recvCounts.data(), displs.data(),
        MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    return out;
}
