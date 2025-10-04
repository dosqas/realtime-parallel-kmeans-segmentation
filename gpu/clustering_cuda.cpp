#include "clustering_backends.hpp"
#include "coreset.hpp"
#include "utils.hpp"
#include <random>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

__global__ void assignPixelsKernel(
    const uchar3* input,
    uchar3* output,
    int width,
    int height,
    const float* centers, // flattened k*5 array
    int k,
    float color_scale,
    float spatial_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int r = idx / width;
    int c = idx % width;

    float x01 = float(c) / float(width);
    float y01 = float(r) / float(height);

    uchar3 pix = input[idx];
    float f[5] = {
        pix.x * color_scale,
        pix.y * color_scale,
        pix.z * color_scale,
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

    uchar3 color;
    color.x = static_cast<uchar>(centers[bestIdx * 5 + 0] / std::max(1e-6f, color_scale));
    color.y = static_cast<uchar>(centers[bestIdx * 5 + 1] / std::max(1e-6f, color_scale));
    color.z = static_cast<uchar>(centers[bestIdx * 5 + 2] / std::max(1e-6f, color_scale));

    output[idx] = color;
}

cv::Mat segmentFrameWithKMeans_cuda(
    const cv::Mat& frame,
    int k,
    int sample_size,
    float color_scale,
    float spatial_scale)
{
    CV_Assert(frame.type() == CV_8UC3);
    std::vector<cv::Vec<float, 5>> centers = computeKMeansCenters(frame, k, sample_size, color_scale, spatial_scale);

    int rows = frame.rows;
    int cols = frame.cols;
    int totalPixels = rows * cols;

    cv::Mat out(frame.size(), frame.type());

    // Flatten input to uchar3
    uchar3* d_input;
    uchar3* d_output;
    float* d_centers;
    cudaMalloc(&d_input, totalPixels * sizeof(uchar3));
    cudaMalloc(&d_output, totalPixels * sizeof(uchar3));
    cudaMalloc(&d_centers, k * 5 * sizeof(float));

    // Copy input frame
    cudaMemcpy(d_input, frame.ptr<uchar3>(), totalPixels * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Flatten centers
    std::vector<float> flatCenters(k * 5);
    for (int i = 0; i < k; ++i)
        for (int d = 0; d < 5; ++d)
            flatCenters[i * 5 + d] = centers[i][d];

    cudaMemcpy(d_centers, flatCenters.data(), k * 5 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    assignPixelsKernel << <blocks, threadsPerBlock >> > (d_input, d_output, cols, rows, d_centers, k, color_scale, spatial_scale);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(out.ptr<uchar3>(), d_output, totalPixels * sizeof(uchar3), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_centers);

    return out;
}
