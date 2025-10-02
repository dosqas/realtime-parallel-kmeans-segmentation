#pragma once
#include <opencv2/core.hpp>

// A point in a coreset represents a compressed version of one (when we sample the frame initially) or more pixels 
// (when we merge coresets).
// - `rgb`: the average RGB color of the sampled pixel(s).
// - `x, y`: the normalized spatial coordinates of the pixel in the frame ([0, 1] range).
// - `weight`: how many original pixels this point represents.
//    (e.g., if we sampled 1,000 out of 1,000,000 pixels, each sampled point has weight ~1000).
struct CoresetPoint {
    cv::Vec3f rgb;
    float x, y;
    float weight;
};

// A coreset is a small, weighted subset of points that approximates the original dataset.
// In this case, it compresses tens or hundreds of thousands of pixels into only a few thousand weighted points.
// 
// Benefits:
// - Performance: running k-means on a coreset is much faster than on all pixels.
// - Accuracy: theory guarantees that clustering on a coreset gives results within (1±ε) of the full dataset,
//             and in practice it can even improve quality by reducing noise and redundancy.
// - Mergeability: coresets can be combined (merged) into new coresets while keeping size bounded.
struct Coreset {
    std::vector<CoresetPoint> points;
};

// Build a coreset by randomly sampling `sample_size` pixels from the frame
// and creating weighted points that summarize them.
// 
// Args:
//   frame: input image (cv::Mat, 3-channel BGR)
//   sample_size: how many pixels to randomly sample for coreset
//
// Returns:
//   A Coreset with `sample_size` points, each with appropriate weights.
Coreset buildCoresetFromFrame(const cv::Mat& frame, int sample_size);

// Merge two coresets into one, keeping the size bounded by `sample_size`.
// The result is still a valid coreset approximating the union of A and B.
// 
// Args:
//   A, B: input coresets to merge
//   sample_size: how many pixels to randomly sample for coreset
//
// Returns:
//   A merged Coreset with `sample_size` points.
Coreset mergeCoresets(const Coreset& A, const Coreset& B, int sample_size);