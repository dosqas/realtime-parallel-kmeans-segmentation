#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "clustering_backends.hpp"
#include "coreset.hpp"
#include "utils.hpp"
#include <random>
#include <opencv2/core.hpp>
#include <iostream>

// CUDA kernel to assign each pixel to the nearest cluster center
//
// Args:
//   input: Pointer to input image data (BGR format)
//   output: Pointer to output image data (BGR format)
//   width: Width of the image
//   height: Height of the image
//   centers: Pointer to cluster centers (k x 5 array)
//   k: Number of clusters
//   color_scale: Scaling factor for color features
//   spatial_scale: Scaling factor for spatial features
__global__ static void assignPixelsKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    const float* centers,
    int k,
    float color_scale,
    float spatial_scale)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // Linear index of the pixel based on block and thread indices
    int total = width * height;
    if (idx >= total) return;

    int r = idx / width;
    int c = idx % width;
    int offset = idx * 3;

	float x01 = float(c) / float(width);  // Normalize x coordinate to [0, 1]
	float y01 = float(r) / float(height); // Normalize y coordinate to [0, 1]

	// Create feature vector and scale color and spatial features
    float f[5] = {
        (float)input[offset + 0] * color_scale,  // B 
        (float)input[offset + 1] * color_scale,  // G
        (float)input[offset + 2] * color_scale,  // R
        x01 * spatial_scale,
        y01 * spatial_scale
    };

    int bestIdx = 0;
	float bestDist2 = 1e20f; // Initialize with a large value

	// Find the nearest cluster center
    for (int ci = 0; ci < k; ++ci) {
        float d2 = 0.0f;
        
		// Compute squared Euclidean distance in 5D space (BGRXY)
        for (int d = 0; d < 5; ++d) {
            float diff = f[d] - centers[ci * 5 + d];
            d2 += diff * diff;
        }
        if (d2 < bestDist2) {
            bestDist2 = d2;
            bestIdx = ci;
        }
    }

    float inv_scale = 1.0f / fmaxf(1e-6f, color_scale);
    // Determine the output pixel color as the color of the nearest center, scaled back by the color_scale factor
    output[offset + 0] = (unsigned char)fminf(255.0f, centers[bestIdx * 5 + 0] * inv_scale); // B
    output[offset + 1] = (unsigned char)fminf(255.0f, centers[bestIdx * 5 + 1] * inv_scale); // G
    output[offset + 2] = (unsigned char)fminf(255.0f, centers[bestIdx * 5 + 2] * inv_scale); // R
}

// Segment a frame using K-means clustering (CUDA implementation)
cv::Mat segmentFrameWithKMeans_cuda(
    const cv::Mat& frame,
    int k,
    int sample_size,
    float color_scale,
    float spatial_scale)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

	// Check for CUDA errors
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return frame.clone(); // Return original frame
    }

    CV_Assert(frame.type() == CV_8UC3); // Ensure input is 3-channel BGR image

    // Compute cluster centers on host
    std::vector<cv::Vec<float, 5>> centers = computeKMeansCenters(frame, k, sample_size, color_scale, spatial_scale);

    int rows = frame.rows;
    int cols = frame.cols;
    int totalPixels = rows * cols;
    cv::Mat out(frame.size(), frame.type());

    // Allocate device memory
    unsigned char* d_input;
    unsigned char* d_output;
    float* d_centers;

    // Allocate for the 3 channels of the input frame
    cudaMalloc(&d_input, static_cast<unsigned long long>(totalPixels) * 3 * sizeof(unsigned char));

    // Allocate for the 3 channels of the output frame
    cudaMalloc(&d_output, static_cast<unsigned long long>(totalPixels) * 3 * sizeof(unsigned char));

    // Each center has 5 dimensions (B, G, R, x, y)
    cudaMalloc(&d_centers, static_cast<unsigned long long>(k) * 5 * sizeof(float));

	// Copy input frame to device (total pixels * 3 channels)
    cudaMemcpy(d_input, frame.data, static_cast<unsigned long long>(totalPixels) * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

	// Flatten centers array to improve memory transfer
    std::vector<float> flatCenters(k * 5);
    for (int i = 0; i < k; ++i)
        for (int d = 0; d < 5; ++d)
            flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(i) * 5 + d] = centers[i][d];

	// Copy centers to device
    cudaMemcpy(d_centers, flatCenters.data(), static_cast<unsigned long long>(k) * 5 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
	// Calculate number of blocks needed (equals totalPixels / threadsPerBlock, rounded up)
    int blocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

	// Launch the kernel to assign pixels to nearest cluster center
	// IntelliSense may suggest that the code below has an error, but it is correct.
	// It is a known issue with IntelliSense not fully understanding CUDA syntax.
	// It is safe to ignore the warning in this case
    assignPixelsKernel <<<blocks, threadsPerBlock >>> (d_input, d_output, cols, rows, d_centers, k, color_scale, spatial_scale);

	// Check for kernel launch errors and synchronize
    cudaGetLastError();
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(out.data, d_output, static_cast<unsigned long long>(totalPixels) * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_centers);

    return out;
}