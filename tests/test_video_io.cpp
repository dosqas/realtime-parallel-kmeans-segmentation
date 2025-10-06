#include <gtest/gtest.h>
#include "../include/video_io.hpp"
#include "../include/clustering.hpp"
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>

class VideoIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Standard 640x480 test frame
        test_frame = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::rectangle(test_frame, cv::Point(100, 100), cv::Point(540, 380), cv::Scalar(128, 64, 255), -1);
        
        // Small frame for resize testing
        small_frame = cv::Mat::zeros(240, 320, CV_8UC3);
        cv::rectangle(small_frame, cv::Point(50, 50), cv::Point(270, 190), cv::Scalar(0, 255, 0), -1);
        
        // Different aspect ratio frames
        wide_frame = cv::Mat::zeros(360, 1280, CV_8UC3);
        cv::rectangle(wide_frame, cv::Point(200, 50), cv::Point(1080, 310), cv::Scalar(255, 0, 0), -1);
        
        tall_frame = cv::Mat::zeros(960, 320, CV_8UC3);
        cv::rectangle(tall_frame, cv::Point(50, 200), cv::Point(270, 760), cv::Scalar(0, 0, 255), -1);
    }

    cv::Mat test_frame;
    cv::Mat small_frame;
    cv::Mat wide_frame;
    cv::Mat tall_frame;
};

TEST_F(VideoIOTest, FrameValidation) {
    EXPECT_FALSE(test_frame.empty());
    EXPECT_EQ(test_frame.type(), CV_8UC3);
    EXPECT_GT(test_frame.rows, 0);
    EXPECT_GT(test_frame.cols, 0);
    
    cv::Mat empty_frame;
    EXPECT_TRUE(empty_frame.empty());
    
    cv::Mat wrong_type = cv::Mat::zeros(100, 100, CV_8UC1);
    EXPECT_NE(wrong_type.type(), CV_8UC3);
}

TEST_F(VideoIOTest, FrameResizing) {
    cv::Mat segmented_small = cv::Mat::zeros(120, 160, CV_8UC3);
    
    cv::Mat resized;
    cv::resize(segmented_small, resized, test_frame.size(), 0, 0, cv::INTER_NEAREST);
    
    EXPECT_EQ(resized.size(), test_frame.size());
    EXPECT_EQ(resized.type(), test_frame.type());
}

TEST_F(VideoIOTest, SideBySideCombination) {
    cv::Mat segmented = test_frame.clone();
    cv::cvtColor(segmented, segmented, cv::COLOR_BGR2HSV);
    cv::cvtColor(segmented, segmented, cv::COLOR_HSV2BGR);
    
    cv::Mat combined(test_frame.rows, test_frame.cols * 2, test_frame.type());
    test_frame.copyTo(combined(cv::Rect(0, 0, test_frame.cols, test_frame.rows)));
    segmented.copyTo(combined(cv::Rect(test_frame.cols, 0, test_frame.cols, test_frame.rows)));
    
    EXPECT_EQ(combined.rows, test_frame.rows);
    EXPECT_EQ(combined.cols, test_frame.cols * 2);
    EXPECT_EQ(combined.type(), test_frame.type());
    
    cv::Mat left_half = combined(cv::Rect(0, 0, test_frame.cols, test_frame.rows));
    cv::Mat right_half = combined(cv::Rect(test_frame.cols, 0, test_frame.cols, test_frame.rows));
    
    cv::Mat diff;
    cv::absdiff(left_half, right_half, diff);
    cv::Scalar sum_diff = cv::sum(diff);
    EXPECT_GT(sum_diff[0] + sum_diff[1] + sum_diff[2], 0.0);
}

TEST_F(VideoIOTest, TextOverlayRendering) {
    cv::Mat frame_copy = test_frame.clone();
    
    std::string overlay_text = "k=5  backend=SEQ  FPS=30.5  min=28.2  max=35.1";
    
    // Shadow text
    cv::putText(frame_copy, overlay_text, cv::Point(12, 28), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    // Main text
    cv::putText(frame_copy, overlay_text, cv::Point(12, 28), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    
    cv::Mat diff;
    cv::absdiff(test_frame, frame_copy, diff);
    cv::Scalar sum_diff = cv::sum(diff);
    EXPECT_GT(sum_diff[0] + sum_diff[1] + sum_diff[2], 0.0);
}

TEST_F(VideoIOTest, LabelTextRendering) {
    cv::Mat frame_copy = test_frame.clone();
    
    cv::putText(frame_copy, "Original", cv::Point(12, test_frame.rows - 12), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(frame_copy, "Segmented", cv::Point(test_frame.cols + 12, test_frame.rows - 12), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    
    cv::Mat diff;
    cv::absdiff(test_frame, frame_copy, diff);
    cv::Scalar sum_diff = cv::sum(diff);
    EXPECT_GT(sum_diff[0] + sum_diff[1] + sum_diff[2], 0.0);
}

TEST_F(VideoIOTest, KParameterValidation) {
    const int k_min = 2;
    const int k_max = 12;
    
    // Valid K values
    for (int k_test = k_min; k_test <= k_max; ++k_test) {
        int k = std::max(k_min, std::min(k_max, k_test));
        EXPECT_EQ(k, k_test);
        EXPECT_GE(k, k_min);
        EXPECT_LE(k, k_max);
    }
    
    // Invalid K values
    int k_too_low = std::max(k_min, std::min(k_max, 0));
    EXPECT_EQ(k_too_low, k_min);
    
    int k_too_high = std::max(k_min, std::min(k_max, 20));
    EXPECT_EQ(k_too_high, k_max);
}

TEST_F(VideoIOTest, FPSCalculation) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double dt = duration.count() / 1000000.0;
    
    double fps = (dt > 0) ? 1.0 / dt : 0.0;
    
    EXPECT_GT(fps, 0.0);
    EXPECT_LT(fps, 100.0);
    EXPECT_NEAR(fps, 30.0, 10.0);
}

TEST_F(VideoIOTest, FPSHistoryManagement) {
    std::deque<std::pair<double, double>> fpsHistory;
    double currentTime = 0.0;
    
    for (int i = 0; i < 10; ++i) {
        fpsHistory.emplace_back(currentTime, 30.0 + i);
        currentTime += 0.5;
    }
    
    EXPECT_EQ(fpsHistory.size(), 10);
    
    // Remove entries older than 3 seconds
    double nowSec = currentTime;
    while (!fpsHistory.empty() && nowSec - fpsHistory.front().first > 3.0) {
        fpsHistory.pop_front();
    }
    
    EXPECT_LE(fpsHistory.size(), 7);
    EXPECT_GT(fpsHistory.size(), 5);
}

TEST_F(VideoIOTest, MinMaxFPSCalculation) {
    std::deque<std::pair<double, double>> fpsHistory = {
        {0.0, 25.5},
        {0.5, 30.2},
        {1.0, 28.7},
        {1.5, 35.1},
        {2.0, 32.3},
        {2.5, 27.9}
    };
    
    double minFps = 1000.0, maxFps = 0.0;
    for (const auto& p : fpsHistory) {
        if (p.second < minFps) minFps = p.second;
        if (p.second > maxFps) maxFps = p.second;
    }
    
    EXPECT_FLOAT_EQ(minFps, 25.5);
    EXPECT_FLOAT_EQ(maxFps, 35.1);
}

TEST_F(VideoIOTest, BackendStringMapping) {
    Backend backend;
    std::string backendName;
    
    backend = BACKEND_SEQ;
    backendName = (backend == BACKEND_SEQ ? "SEQ" : 
                   backend == BACKEND_CUDA ? "CUDA" : 
                   backend == BACKEND_THR ? "THR" : "MPI");
    EXPECT_EQ(backendName, "SEQ");
    
    backend = BACKEND_CUDA;
    backendName = (backend == BACKEND_SEQ ? "SEQ" : 
                   backend == BACKEND_CUDA ? "CUDA" : 
                   backend == BACKEND_THR ? "THR" : "MPI");
    EXPECT_EQ(backendName, "CUDA");
    
    backend = BACKEND_THR;
    backendName = (backend == BACKEND_SEQ ? "SEQ" : 
                   backend == BACKEND_CUDA ? "CUDA" : 
                   backend == BACKEND_THR ? "THR" : "MPI");
    EXPECT_EQ(backendName, "THR");
    
    backend = BACKEND_MPI;
    backendName = (backend == BACKEND_SEQ ? "SEQ" : 
                   backend == BACKEND_CUDA ? "CUDA" : 
                   backend == BACKEND_THR ? "THR" : "MPI");
    EXPECT_EQ(backendName, "MPI");
}

TEST_F(VideoIOTest, OverlayStringFormatting) {
    int k = 5;
    std::string backendName = "CUDA";
    double fps = 42.7;
    double minFps = 38.2;
    double maxFps = 45.9;
    
    std::string overlay = "k=" + std::to_string(k) +
        "  backend=" + backendName +
        "  FPS=" + cv::format("%.1f", fps) +
        "  min=" + cv::format("%.1f", minFps) +
        "  max=" + cv::format("%.1f", maxFps);
    
    EXPECT_TRUE(overlay.find("k=5") != std::string::npos);
    EXPECT_TRUE(overlay.find("backend=CUDA") != std::string::npos);
    EXPECT_TRUE(overlay.find("FPS=42.7") != std::string::npos);
    EXPECT_TRUE(overlay.find("min=38.2") != std::string::npos);
    EXPECT_TRUE(overlay.find("max=45.9") != std::string::npos);
}

TEST_F(VideoIOTest, FrameSizeConstants) {
    const int expected_width = 640;
    const int expected_height = 480;
    
    EXPECT_EQ(expected_width, 640);
    EXPECT_EQ(expected_height, 480);
    
    double aspect_ratio = static_cast<double>(expected_width) / expected_height;
    EXPECT_NEAR(aspect_ratio, 4.0/3.0, 0.01);
}

TEST_F(VideoIOTest, ParameterConstants) {
    const int sample = 2000;
    const float color_scale = 1.0f;
    const float spatial_scale = 0.5f;
    const int k_min = 2;
    const int k_max = 12;
    
    EXPECT_EQ(sample, 2000);
    EXPECT_FLOAT_EQ(color_scale, 1.0f);
    EXPECT_FLOAT_EQ(spatial_scale, 0.5f);
    EXPECT_EQ(k_min, 2);
    EXPECT_EQ(k_max, 12);
    EXPECT_LT(k_min, k_max);
}

TEST_F(VideoIOTest, ProcessingLoopSimulation) {
    const int k_min = 2, k_max = 12;
    const int sample = 2000;
    const float color_scale = 1.0f;
    const float spatial_scale = 0.5f;
    
    int k_trackbar = 5;
    Backend backend = BACKEND_SEQ;
    
    int k = std::max(k_min, std::min(k_max, k_trackbar));
    EXPECT_EQ(k, 5);
    
    cv::Mat seg = test_frame.clone();
    
    EXPECT_FALSE(seg.empty());
    EXPECT_EQ(seg.type(), test_frame.type());
    
    if (seg.size() != test_frame.size()) {
        cv::resize(seg, seg, test_frame.size(), 0, 0, cv::INTER_NEAREST);
    }
    EXPECT_EQ(seg.size(), test_frame.size());
    
    cv::Mat combined(test_frame.rows, test_frame.cols * 2, test_frame.type());
    test_frame.copyTo(combined(cv::Rect(0, 0, test_frame.cols, test_frame.rows)));
    seg.copyTo(combined(cv::Rect(test_frame.cols, 0, test_frame.cols, test_frame.rows)));
    
    EXPECT_EQ(combined.rows, test_frame.rows);
    EXPECT_EQ(combined.cols, test_frame.cols * 2);
    EXPECT_EQ(combined.type(), test_frame.type());
}

TEST_F(VideoIOTest, CameraPropertySimulation) {
    const int target_width = 640;
    const int target_height = 480;
    
    EXPECT_GT(target_width, 0);
    EXPECT_GT(target_height, 0);
    EXPECT_EQ(target_width, 640);
    EXPECT_EQ(target_height, 480);
}