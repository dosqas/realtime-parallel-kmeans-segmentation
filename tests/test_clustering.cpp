#include <gtest/gtest.h>
#include "../include/clustering.hpp"
#include "../include/coreset.hpp"
#include "../include/utils.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>

class ClusteringTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 4-region test image for clustering validation
        test_image = cv::Mat::zeros(100, 100, CV_8UC3);
        cv::rectangle(test_image, cv::Point(0, 0), cv::Point(50, 50), cv::Scalar(0, 0, 255), -1);    // Red
        cv::rectangle(test_image, cv::Point(50, 0), cv::Point(100, 50), cv::Scalar(255, 0, 0), -1);  // Blue
        cv::rectangle(test_image, cv::Point(0, 50), cv::Point(50, 100), cv::Scalar(0, 255, 0), -1);  // Green
        cv::rectangle(test_image, cv::Point(50, 50), cv::Point(100, 100), cv::Scalar(255, 255, 255), -1); // White
        
        // Simple 2-region image
        simple_image = cv::Mat::zeros(60, 60, CV_8UC3);
        cv::rectangle(simple_image, cv::Point(0, 0), cv::Point(30, 60), cv::Scalar(0, 0, 255), -1);   // Red
        cv::rectangle(simple_image, cv::Point(30, 0), cv::Point(60, 60), cv::Scalar(255, 255, 255), -1); // White
        
        // Gradient image for complex clustering
        gradient_image = cv::Mat(80, 80, CV_8UC3);
        for (int r = 0; r < 80; ++r) {
            for (int c = 0; c < 80; ++c) {
                uchar blue = static_cast<uchar>(r * 255 / 80);
                uchar green = static_cast<uchar>(c * 255 / 80);
                uchar red = static_cast<uchar>((r + c) * 255 / 160);
                gradient_image.at<cv::Vec3b>(r, c) = cv::Vec3b(blue, green, red);
            }
        }
        
        k = 4;
        sample_size = 1000;
        color_scale = 1.0f;
        spatial_scale = 0.1f;
    }

    cv::Mat test_image;
    cv::Mat simple_image;
    cv::Mat gradient_image;
    int k;
    int sample_size;
    float color_scale;
    float spatial_scale;
};

TEST_F(ClusteringTest, SegmentFrameWithKMeansSequential) {
    cv::Mat result = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_SEQ, color_scale, spatial_scale);
    
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.rows, test_image.rows);
    EXPECT_EQ(result.cols, test_image.cols);
    EXPECT_EQ(result.type(), test_image.type());
}

TEST_F(ClusteringTest, SegmentFrameWithKMeansMultiThreaded) {
    cv::Mat result = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_THR, color_scale, spatial_scale);
    
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.rows, test_image.rows);
    EXPECT_EQ(result.cols, test_image.cols);
    EXPECT_EQ(result.type(), test_image.type());
}

#ifdef ENABLE_MPI
TEST_F(ClusteringTest, SegmentFrameWithKMeansMPI) {
    cv::Mat result = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_MPI, color_scale, spatial_scale);
    
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.rows, test_image.rows);
    EXPECT_EQ(result.cols, test_image.cols);
    EXPECT_EQ(result.type(), test_image.type());
}
#endif

#ifdef ENABLE_CUDA
TEST_F(ClusteringTest, SegmentFrameWithKMeansCUDA) {
    cv::Mat result = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_CUDA, color_scale, spatial_scale);
    
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.rows, test_image.rows);
    EXPECT_EQ(result.cols, test_image.cols);
    EXPECT_EQ(result.type(), test_image.type());
}
#endif

TEST_F(ClusteringTest, SequentialClusteringBasic) {
    cv::Mat result = segmentFrameWithKMeans(simple_image, 2, 500, BACKEND_SEQ, color_scale, spatial_scale);
    
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), simple_image.size());
    
    // Check segmentation created distinct regions
    cv::Mat left_region = result(cv::Rect(0, 0, 15, 60));
    cv::Mat right_region = result(cv::Rect(45, 0, 15, 60));
    
    cv::Scalar left_mean = cv::mean(left_region);
    cv::Scalar right_mean = cv::mean(right_region);
    
    double color_diff = cv::norm(left_mean - right_mean);
    EXPECT_GT(color_diff, 10.0);
}

TEST_F(ClusteringTest, ClusteringParameterValidation) {
    // Invalid k values
    cv::Mat result1 = segmentFrameWithKMeans(test_image, 0, sample_size, BACKEND_SEQ, color_scale, spatial_scale);
    cv::Mat result2 = segmentFrameWithKMeans(test_image, -1, sample_size, BACKEND_SEQ, color_scale, spatial_scale);
    cv::Mat result3 = segmentFrameWithKMeans(test_image, 100, sample_size, BACKEND_SEQ, color_scale, spatial_scale);
    
    // Should handle gracefully without crashing
    EXPECT_FALSE(result3.empty());
}

TEST_F(ClusteringTest, ClusteringScaleParameters) {
    // Different color scales
    cv::Mat result1 = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_SEQ, 0.5f, spatial_scale);
    cv::Mat result2 = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_SEQ, 2.0f, spatial_scale);
    
    EXPECT_FALSE(result1.empty());
    EXPECT_FALSE(result2.empty());
    EXPECT_EQ(result1.size(), result2.size());
    
    // Different spatial scales
    cv::Mat result3 = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_SEQ, color_scale, 0.05f);
    cv::Mat result4 = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_SEQ, color_scale, 0.5f);
    
    EXPECT_FALSE(result3.empty());
    EXPECT_FALSE(result4.empty());
}

TEST_F(ClusteringTest, ClusteringSampleSizeEffect) {
    cv::Mat result_small = segmentFrameWithKMeans(test_image, k, 100, BACKEND_SEQ, color_scale, spatial_scale);
    cv::Mat result_large = segmentFrameWithKMeans(test_image, k, 2000, BACKEND_SEQ, color_scale, spatial_scale);
    
    EXPECT_FALSE(result_small.empty());
    EXPECT_FALSE(result_large.empty());
    EXPECT_EQ(result_small.size(), result_large.size());
    EXPECT_EQ(result_small.type(), CV_8UC3);
    EXPECT_EQ(result_large.type(), CV_8UC3);
}

TEST_F(ClusteringTest, BackendConsistency) {
    cv::Mat seq_result = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_SEQ, color_scale, spatial_scale);
    cv::Mat thr_result = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_THR, color_scale, spatial_scale);
    
    EXPECT_EQ(seq_result.size(), thr_result.size());
    EXPECT_EQ(seq_result.type(), thr_result.type());
    
    // Results should be reasonably bounded (allowing for random differences)
    cv::Mat diff;
    cv::absdiff(seq_result, thr_result, diff);
    cv::Scalar mean_diff = cv::mean(diff);
    
    EXPECT_LT(mean_diff[0], 100.0);
    EXPECT_LT(mean_diff[1], 100.0);
    EXPECT_LT(mean_diff[2], 100.0);
}

TEST_F(ClusteringTest, ClusteringReproducibility) {
    std::vector<cv::Mat> results;
    
    for (int trial = 0; trial < 3; ++trial) {
        cv::Mat result = segmentFrameWithKMeans(simple_image, 2, 500, BACKEND_SEQ, color_scale, spatial_scale);
        results.push_back(result);
        
        EXPECT_FALSE(result.empty());
        EXPECT_EQ(result.size(), simple_image.size());
    }
    
    // Consistent dimensions and type across runs
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_EQ(results[0].size(), results[i].size());
        EXPECT_EQ(results[0].type(), results[i].type());
    }
}

TEST_F(ClusteringTest, ClusteringPerformanceBenchmark) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cv::Mat result = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_SEQ, color_scale, spatial_scale);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    EXPECT_FALSE(result.empty());
    EXPECT_LT(duration.count(), 5000); // Complete within 5 seconds
    
    double fps = 1000.0 / duration.count();
    EXPECT_GT(fps, 0.2); // Minimum performance threshold
}

TEST_F(ClusteringTest, ClusteringMemoryStability) {
    // Multiple consecutive operations for memory leak detection
    for (int i = 0; i < 10; ++i) {
        cv::Mat result = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_SEQ, color_scale, spatial_scale);
        
        EXPECT_FALSE(result.empty());
        EXPECT_EQ(result.size(), test_image.size());
        
        // Verify no NaN values
        cv::Scalar mean_color = cv::mean(result);
        EXPECT_FALSE(std::isnan(mean_color[0]));
        EXPECT_FALSE(std::isnan(mean_color[1]));
        EXPECT_FALSE(std::isnan(mean_color[2]));
    }
}

TEST_F(ClusteringTest, ClusteringComplexImage) {
    cv::Mat result = segmentFrameWithKMeans(gradient_image, k, sample_size, BACKEND_SEQ, color_scale, spatial_scale);
    
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), gradient_image.size());
    EXPECT_EQ(result.type(), gradient_image.type());
    
    // Check color variation in result
    cv::Mat result_gray;
    cv::cvtColor(result, result_gray, cv::COLOR_BGR2GRAY);
    
    double min_val, max_val;
    cv::minMaxLoc(result_gray, &min_val, &max_val);
    
    EXPECT_LT(max_val - min_val, 255.0);
    EXPECT_GT(max_val - min_val, 10.0);
}

TEST_F(ClusteringTest, ClusteringEdgeCases) {
    // Very small image
    cv::Mat tiny_image = cv::Mat::zeros(5, 5, CV_8UC3);
    cv::rectangle(tiny_image, cv::Point(0, 0), cv::Point(2, 5), cv::Scalar(255, 0, 0), -1);
    cv::rectangle(tiny_image, cv::Point(3, 0), cv::Point(5, 5), cv::Scalar(0, 255, 0), -1);
    
    cv::Mat result = segmentFrameWithKMeans(tiny_image, 2, 20, BACKEND_SEQ, color_scale, spatial_scale);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), tiny_image.size());
    
    // Uniform color image
    cv::Mat uniform_image = cv::Mat(50, 50, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat uniform_result = segmentFrameWithKMeans(uniform_image, k, sample_size, BACKEND_SEQ, color_scale, spatial_scale);
    EXPECT_FALSE(uniform_result.empty());
    EXPECT_EQ(uniform_result.size(), uniform_image.size());
}

TEST_F(ClusteringTest, ClusteringBackendSwitching) {
    std::vector<Backend> backends = {BACKEND_SEQ, BACKEND_THR};
    
#ifdef ENABLE_MPI
    backends.push_back(BACKEND_MPI);
#endif

#ifdef ENABLE_CUDA
    backends.push_back(BACKEND_CUDA);
#endif
    
    std::vector<cv::Mat> results;
    for (Backend backend : backends) {
        cv::Mat result = segmentFrameWithKMeans(test_image, k, sample_size, backend, color_scale, spatial_scale);
        
        EXPECT_FALSE(result.empty()) << "Backend failed: " << static_cast<int>(backend);
        EXPECT_EQ(result.size(), test_image.size());
        EXPECT_EQ(result.type(), test_image.type());
        
        results.push_back(result);
    }
    
    // Consistent dimensions across backends
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_EQ(results[0].size(), results[i].size());
        EXPECT_EQ(results[0].type(), results[i].type());
    }
}

TEST_F(ClusteringTest, BackendPerformanceComparison) {
    struct BackendTiming {
        Backend backend;
        std::string name;
        double time_ms;
    };
    
    std::vector<BackendTiming> timings;
    
    // Sequential backend
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat result = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_SEQ, color_scale, spatial_scale);
        auto end = std::chrono::high_resolution_clock::now();
        
        EXPECT_FALSE(result.empty());
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back({BACKEND_SEQ, "Sequential", time_ms});
    }
    
    // Multithreaded backend
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat result = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_THR, color_scale, spatial_scale);
        auto end = std::chrono::high_resolution_clock::now();
        
        EXPECT_FALSE(result.empty());
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back({BACKEND_THR, "Multithreaded", time_ms});
    }
    
#ifdef ENABLE_CUDA
    // CUDA backend
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat result = segmentFrameWithKMeans(test_image, k, sample_size, BACKEND_CUDA, color_scale, spatial_scale);
        auto end = std::chrono::high_resolution_clock::now();
        
        EXPECT_FALSE(result.empty());
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back({BACKEND_CUDA, "CUDA", time_ms});
    }
#endif
    
    // Performance results
    std::cout << "\n=== Backend Performance Comparison ===" << std::endl;
    for (const auto& timing : timings) {
        double fps = 1000.0 / timing.time_ms;
        std::cout << timing.name << ": " << timing.time_ms << " ms (" << fps << " FPS)" << std::endl;
        
        EXPECT_LT(timing.time_ms, 10000.0); // Under 10 seconds
        EXPECT_GT(fps, 0.1); // Minimum FPS
    }
}