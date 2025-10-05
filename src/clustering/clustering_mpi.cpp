#include "clustering_backends.hpp"
#include "coreset.hpp"
#include "utils.hpp"
#include <random>
#include <mpi.h>

// Segment a frame using K-means clustering (distributed implementation)
// Uses Message Passing Interface (MPI) for distributed processing across multiple processes/nodes
cv::Mat segmentFrameWithKMeans_mpi(
    const cv::Mat& frame,
    int k,
    int sample_size,
    float color_scale,
    float spatial_scale)
{
    CV_Assert(frame.type() == CV_8UC3); // Ensure input is 3-channel BGR image

	// Ranks represent different processes in MPI
	// A process with rank 0 is typically the master process
    int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
	MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

	std::vector<float> flatCenters; // Flattened K-means centers for broadcasting (improves MPI performance)
	// Only rank 0 computes the centers
    if (rank == 0) {
        auto centers = computeKMeansCenters(frame, k, sample_size, color_scale, spatial_scale);
		// Each center is a 5D vector (BGRXY), so we flatten them into a single array for MPI
		// We use resize with static_cast to avoid warnings about signed/unsigned comparison
        flatCenters.resize(static_cast<std::vector<float, std::allocator<float>>::size_type>(k) * 5); 

		// For each center and each of its 5 dimensions, copy to the flat array
        for (int i = 0; i < k; ++i)
            for (int d = 0; d < 5; ++d)
                flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(i) * 5 + d] = centers[i][d];
    }

	// Broadcast all centers in one go to all ranks
    if (rank != 0) flatCenters.resize(static_cast<std::vector<float, std::allocator<float>>::size_type>(k) * 5);
    MPI_Bcast(flatCenters.data(), k * 5, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int rows = frame.rows;
    int cols = frame.cols;

	// Divide work by rows among ranks
	// Last rank may take extra rows if rows is not divisible by size
    int rowsPerRank = rows / size;
    int startRow = rank * rowsPerRank;
    int endRow = (rank == size - 1) ? rows : startRow + rowsPerRank;

    cv::Mat localOut(endRow - startRow, cols, frame.type());

    // Each rank processes its chunk in parallel
	// The pragma omp directive enables multi-threading within each rank
#pragma omp parallel for schedule(dynamic)
	// First go row by row in the assigned chunk
    for (int r = startRow; r < endRow; ++r) {
		const cv::Vec3b* inRow = frame.ptr<cv::Vec3b>(r); // Pointer to the current row in input image
		cv::Vec3b* outRow = localOut.ptr<cv::Vec3b>(r - startRow); // Pointer to the current row in output image
		float y01 = (float)r / (float)rows; // Normalized y coordinate

		// Then go pixel by pixel in the row
        for (int c = 0; c < cols; ++c) {
			const cv::Vec3b& pix = inRow[c]; // Current pixel color
			float x01 = (float)c / (float)cols; // Normalized x coordinate

            float f[5] = {
                pix[0] * color_scale, // B
                pix[1] * color_scale, // G
                pix[2] * color_scale, // R
                x01 * spatial_scale,
                y01 * spatial_scale
            };

            int bestIdx = 0;
            float bestDist2 = std::numeric_limits<float>::max();

			// Find the nearest K-means center to the pixel's feature vector by going through all centers
            for (int ci = 0; ci < k; ++ci) {
                float d2 = 0.f;
				// The pragma unroll directive suggests the compiler to unroll the loop for performance
#pragma unroll
				// Compute squared Euclidean distance in 5D space for each of the 5 dimensions (BGRXY)
                for (int d = 0; d < 5; ++d) {
                    float diff = f[d] - flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(ci) * 5 + d];
                    d2 += diff * diff;
                }
                if (d2 < bestDist2) { bestDist2 = d2; bestIdx = ci; }
            }

            cv::Vec3b color;
            // Determine the output pixel color as the color of the nearest center, scaled back by the color_scale factor
            color[0] = (uchar)std::min(255.f, 
                flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(bestIdx) * 5 + 0] / 
                std::max(1e-6f, color_scale)); // B
            color[1] = (uchar)std::min(255.f, 
                flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(bestIdx) * 5 + 1] / 
                std::max(1e-6f, color_scale)); // G
            color[2] = (uchar)std::min(255.f, 
                flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(bestIdx) * 5 + 2] / 
                std::max(1e-6f, color_scale)); // R
            outRow[c] = color;
        }
    }

    // Gather results at rank 0
    cv::Mat out;
    if (rank == 0) {
		out.create(rows, cols, frame.type()); // Only rank 0 creates the final output image
    }

	// Prepare counts and displacements for MPI_Gatherv
	// recvCounts[i] = number of elements received from rank i
	// displs[i] = displacement (offset) in the final array where data from rank i should be placed
    std::vector<int> recvCounts(size), displs(size);
	// Each rank contributes its chunk of rows
    for (int i = 0; i < size; ++i) {
		// Calculate start and end rows for each rank
        int sRow = i * rowsPerRank;
        int eRow = (i == size - 1) ? rows : sRow + rowsPerRank;
		recvCounts[i] = (eRow - sRow) * cols * 3; // Each pixel has 3 channels (BGR)
		displs[i] = sRow * cols * 3; // Displacement in the final array
    }

	// Gather all local outputs into the final output at rank 0
    MPI_Gatherv(localOut.data,
        (endRow - startRow) * cols * 3, MPI_UNSIGNED_CHAR,
        out.data, recvCounts.data(), displs.data(),
        MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    return out;
}
