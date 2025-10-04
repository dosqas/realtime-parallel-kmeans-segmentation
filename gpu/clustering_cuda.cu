#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "clustering_backends.hpp"
#include "coreset.hpp"
#include "utils.hpp"
#include <random>
#include <opencv2/core.hpp>
#include <iostream>

__global__ void assignPixelsKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    const float* centers,
    int k,
    float color_scale,
    float spatial_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int r = idx / width;
    int c = idx % width;
    int offset = idx * 3;

    float x01 = float(c) / float(width);
    float y01 = float(r) / float(height);

    // Create feature vector - scale colors like sequential version does
    float f[5] = {
        (float)input[offset + 0] * color_scale,  // B (not R!)
        (float)input[offset + 1] * color_scale,  // G
        (float)input[offset + 2] * color_scale,  // R
        x01 * spatial_scale,
        y01 * spatial_scale
    };

    int bestIdx = 0;
    float bestDist2 = 1e20f;

    for (int ci = 0; ci < k; ++ci) {
        float d2 = 0.0f;
        for (int d = 0; d < 5; ++d) {
            float diff = f[d] - centers[ci * 5 + d];
            d2 += diff * diff;
        }
        if (d2 < bestDist2) {
            bestDist2 = d2;
            bestIdx = ci;
        }
    }

    // Write output - divide by color_scale like sequential version
    float inv_scale = 1.0f / fmaxf(1e-6f, color_scale);
    output[offset + 0] = (unsigned char)fminf(255.0f, centers[bestIdx * 5 + 0] * inv_scale); // B
    output[offset + 1] = (unsigned char)fminf(255.0f, centers[bestIdx * 5 + 1] * inv_scale); // G
    output[offset + 2] = (unsigned char)fminf(255.0f, centers[bestIdx * 5 + 2] * inv_scale); // R
}

cv::Mat segmentFrameWithKMeans_cuda(
    const cv::Mat& frame,
    int k,
    int sample_size,
    float color_scale,
    float spatial_scale)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA devices: " << deviceCount << ", status: " << cudaGetErrorString(err) << std::endl;

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return frame.clone(); // Return original frame
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << std::endl;

    CV_Assert(frame.type() == CV_8UC3);

    // Compute cluster centers on host
    std::vector<cv::Vec<float, 5>> centers = computeKMeansCenters(frame, k, sample_size, color_scale, spatial_scale);

    std::cout << "=== DEBUG INFO ===" << std::endl;
    std::cout << "Frame: " << frame.cols << "x" << frame.rows << std::endl;
    std::cout << "First input pixel BGR: "
        << (int)frame.data[0] << ", "
        << (int)frame.data[1] << ", "
        << (int)frame.data[2] << std::endl;

    int rows = frame.rows;
    int cols = frame.cols;
    int totalPixels = rows * cols;
    cv::Mat out(frame.size(), frame.type());

    // Initialize output to something visible (for testing)
    out = cv::Scalar(255, 0, 0); // Blue - if you see this, kernel didn't run

    // Allocate device memory
    unsigned char* d_input;
    unsigned char* d_output;
    float* d_centers;

    err = cudaMalloc(&d_input, totalPixels * 3 * sizeof(unsigned char));
    std::cout << "cudaMalloc d_input: " << cudaGetErrorString(err) << std::endl;

    err = cudaMalloc(&d_output, totalPixels * 3 * sizeof(unsigned char));
    std::cout << "cudaMalloc d_output: " << cudaGetErrorString(err) << std::endl;

    err = cudaMalloc(&d_centers, k * 5 * sizeof(float));
    std::cout << "cudaMalloc d_centers: " << cudaGetErrorString(err) << std::endl;

    // Copy input frame to device
    err = cudaMemcpy(d_input, frame.data, totalPixels * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    std::cout << "cudaMemcpy input: " << cudaGetErrorString(err) << std::endl;

    // Flatten centers array
    std::vector<float> flatCenters(k * 5);
    for (int i = 0; i < k; ++i)
        for (int d = 0; d < 5; ++d)
            flatCenters[i * 5 + d] = centers[i][d];

    err = cudaMemcpy(d_centers, flatCenters.data(), k * 5 * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "cudaMemcpy centers: " << cudaGetErrorString(err) << std::endl;

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching kernel: " << blocks << " blocks, " << threadsPerBlock << " threads" << std::endl;

    assignPixelsKernel << <blocks, threadsPerBlock >> > (d_input, d_output, cols, rows, d_centers, k, color_scale, spatial_scale);

    err = cudaGetLastError();
    std::cout << "Kernel launch: " << cudaGetErrorString(err) << std::endl;

    err = cudaDeviceSynchronize();
    std::cout << "Kernel execution: " << cudaGetErrorString(err) << std::endl;

    // Copy result back to host
    err = cudaMemcpy(out.data, d_output, totalPixels * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    std::cout << "cudaMemcpy output: " << cudaGetErrorString(err) << std::endl;

    std::cout << "First output pixel BGR: "
        << (int)out.data[0] << ", "
        << (int)out.data[1] << ", "
        << (int)out.data[2] << std::endl;
    std::cout << "==================" << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_centers);

    return out;
}