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

    std::vector<cv::Vec<float, 5>> centers;

    // Rank 0 computes cluster centers and broadcasts
    if (rank == 0) {
        centers = computeKMeansCenters(frame, k, sample_size, color_scale, spatial_scale);
    }
    // Broadcast centers to all processes
    if (rank == 0) {
        for (int i = 0; i < k; i++) {
            MPI_Bcast(&centers[i][0], 5, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
    }
    else {
        centers.resize(k);
        for (int i = 0; i < k; i++) {
            MPI_Bcast(&centers[i][0], 5, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
    }

    int rows = frame.rows;
    int cols = frame.cols;

    // Divide work (block of rows per process)
    int rowsPerRank = rows / size;
    int startRow = rank * rowsPerRank;
    int endRow = (rank == size - 1) ? rows : startRow + rowsPerRank;

    // Each rank computes its chunk
    cv::Mat localOut(endRow - startRow, cols, frame.type());
    for (int r = startRow; r < endRow; ++r) {
        const cv::Vec3b* inRow = frame.ptr<cv::Vec3b>(r);
        cv::Vec3b* outRow = localOut.ptr<cv::Vec3b>(r - startRow);
        float y01 = (float)r / (float)rows;
        for (int c = 0; c < cols; ++c) {
            const cv::Vec3b& pix = inRow[c];
            float x01 = (float)c / (float)cols;
            cv::Vec<float, 5> f = makeFeature(cv::Vec3f(pix[0], pix[1], pix[2]),
                x01, y01, color_scale, spatial_scale);

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
            color[0] = (uchar)cv::saturate_cast<uchar>(
                centers[bestIdx][0] / std::max(1e-6f, color_scale));
            color[1] = (uchar)cv::saturate_cast<uchar>(
                centers[bestIdx][1] / std::max(1e-6f, color_scale));
            color[2] = (uchar)cv::saturate_cast<uchar>(
                centers[bestIdx][2] / std::max(1e-6f, color_scale));
            outRow[c] = color;
        }
    }

    // Gather results at rank 0
    cv::Mat out;
    if (rank == 0) {
        out.create(rows, cols, frame.type());
    }

    // Gather row counts
    std::vector<int> recvCounts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
        int sRow = i * rowsPerRank;
        int eRow = (i == size - 1) ? rows : sRow + rowsPerRank;
        recvCounts[i] = (eRow - sRow) * cols * 3; // bytes per pixel
        displs[i] = sRow * cols * 3;
    }

    MPI_Gatherv(localOut.data,
        (endRow - startRow) * cols * 3, MPI_UNSIGNED_CHAR,
        out.data, recvCounts.data(), displs.data(),
        MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    return out;
}
