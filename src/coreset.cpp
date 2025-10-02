#include "coreset.hpp"
#include <opencv2/opencv.hpp>
#include <random>
#include <iostream>

Coreset buildCoresetFromFrame(const cv::Mat& frame, int sample_size) {
    Coreset coreset;

    int rows = frame.rows;
    int cols = frame.cols;
    int total_pixels = rows * cols;

    if (sample_size > total_pixels) sample_size = total_pixels;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> row_dist(0, rows - 1);
    std::uniform_int_distribution<> col_dist(0, cols - 1);

    for (int i = 0; i < sample_size; ++i) {
        int r = row_dist(gen);
        int c = col_dist(gen);
        cv::Vec3b pixel = frame.at<cv::Vec3b>(r, c);

        CoresetPoint pt;
        pt.rgb = cv::Vec3f(pixel[0], pixel[1], pixel[2]);
        pt.weight = float(total_pixels) / float(sample_size);
		pt.x = float(c) / float(cols);
		pt.y = float(r) / float(rows);

        coreset.points.push_back(pt);
    }

    return coreset;
}

Coreset mergeCoresets(const Coreset& A, const Coreset& B, int sample_size) {
    Coreset merged;

    merged.points.insert(merged.points.end(), A.points.begin(), A.points.end());
    merged.points.insert(merged.points.end(), B.points.begin(), B.points.end());

    if (merged.points.size() > sample_size) {
        std::shuffle(merged.points.begin(), merged.points.end(), std::mt19937{ std::random_device{}() });
        merged.points.resize(sample_size);
    }

    return merged;
}
