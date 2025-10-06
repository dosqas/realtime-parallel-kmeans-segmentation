#include <gtest/gtest.h>
#include "../include/coreset.hpp"
#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <algorithm>

class CoresetTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Uniform purple image for basic testing
        uniform_image = cv::Mat(100, 100, CV_8UC3, cv::Scalar(128, 64, 255));
        
        // 2x2 image with 4 distinct colors
        tiny_image = cv::Mat(2, 2, CV_8UC3);
        tiny_image.at<cv::Vec3b>(0, 0) = cv::Vec3b(255, 0, 0);   // Blue
        tiny_image.at<cv::Vec3b>(0, 1) = cv::Vec3b(0, 255, 0);   // Green
        tiny_image.at<cv::Vec3b>(1, 0) = cv::Vec3b(0, 0, 255);   // Red
        tiny_image.at<cv::Vec3b>(1, 1) = cv::Vec3b(255, 255, 255); // White
        
        // Checkerboard pattern for spatial testing
        checkerboard = cv::Mat::zeros(50, 50, CV_8UC3);
        for (int r = 0; r < 50; ++r) {
            for (int c = 0; c < 50; ++c) {
                if ((r / 10 + c / 10) % 2 == 0) {
                    checkerboard.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
                } else {
                    checkerboard.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
                }
            }
        }
        
        // Gradient image for complex color distribution
        gradient_image = cv::Mat(100, 100, CV_8UC3);
        for (int r = 0; r < 100; ++r) {
            for (int c = 0; c < 100; ++c) {
                uchar val = static_cast<uchar>((r + c) * 255 / 200);
                gradient_image.at<cv::Vec3b>(r, c) = cv::Vec3b(val, val, val);
            }
        }
    }

    cv::Mat uniform_image;
    cv::Mat tiny_image;
    cv::Mat checkerboard;
    cv::Mat gradient_image;
};

TEST_F(CoresetTest, BuildCoresetBasicFunctionality) {
    int sample_size = 500;
    Coreset coreset = buildCoresetFromFrame(uniform_image, sample_size);
    
    EXPECT_EQ(coreset.points.size(), sample_size);
    
    // Verify BGR values match uniform image
    cv::Vec3f expected_bgr(128.0f, 64.0f, 255.0f);
    for (const auto& point : coreset.points) {
        EXPECT_FLOAT_EQ(point.bgr[0], expected_bgr[0]);
        EXPECT_FLOAT_EQ(point.bgr[1], expected_bgr[1]);
        EXPECT_FLOAT_EQ(point.bgr[2], expected_bgr[2]);
    }
}

TEST_F(CoresetTest, BuildCoresetSampleSizeConstraints) {
    int total_pixels = tiny_image.rows * tiny_image.cols; // 4 pixels
    
    // Sample size larger than available pixels
    Coreset coreset1 = buildCoresetFromFrame(tiny_image, 100);
    EXPECT_EQ(coreset1.points.size(), total_pixels);
    
    // Sample size equal to available pixels
    Coreset coreset2 = buildCoresetFromFrame(tiny_image, total_pixels);
    EXPECT_EQ(coreset2.points.size(), total_pixels);
    
    // Sample size smaller than available pixels
    Coreset coreset3 = buildCoresetFromFrame(tiny_image, 2);
    EXPECT_EQ(coreset3.points.size(), 2);
}

TEST_F(CoresetTest, BuildCoresetWeightCalculation) {
    int sample_size = 100;
    int total_pixels = uniform_image.rows * uniform_image.cols;
    float expected_weight = static_cast<float>(total_pixels) / static_cast<float>(sample_size);
    
    Coreset coreset = buildCoresetFromFrame(uniform_image, sample_size);
    
    // Check weight per point
    for (const auto& point : coreset.points) {
        EXPECT_FLOAT_EQ(point.weight, expected_weight);
    }
    
    // Check total weight preservation
    float total_weight = 0.0f;
    for (const auto& point : coreset.points) {
        total_weight += point.weight;
    }
    EXPECT_FLOAT_EQ(total_weight, static_cast<float>(total_pixels));
}

TEST_F(CoresetTest, BuildCoresetCoordinateNormalization) {
    int sample_size = 200;
    Coreset coreset = buildCoresetFromFrame(checkerboard, sample_size);
    
    // Coordinates should be normalized to [0, 1]
    for (const auto& point : coreset.points) {
        EXPECT_GE(point.x, 0.0f);
        EXPECT_LE(point.x, 1.0f);
        EXPECT_GE(point.y, 0.0f);
        EXPECT_LE(point.y, 1.0f);
    }
}

TEST_F(CoresetTest, BuildCoresetRandomness) {
    int sample_size = 100;
    
    // Multiple runs should produce different samples
    Coreset coreset1 = buildCoresetFromFrame(gradient_image, sample_size);
    Coreset coreset2 = buildCoresetFromFrame(gradient_image, sample_size);
    
    bool different = false;
    for (size_t i = 0; i < coreset1.points.size() && i < coreset2.points.size(); ++i) {
        if (coreset1.points[i].x != coreset2.points[i].x || 
            coreset1.points[i].y != coreset2.points[i].y) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

TEST_F(CoresetTest, BuildCoresetZeroSampleSize) {
    Coreset coreset = buildCoresetFromFrame(uniform_image, 0);
    EXPECT_TRUE(coreset.points.empty());
}

TEST_F(CoresetTest, BuildCoresetNegativeSampleSize) {
    Coreset coreset = buildCoresetFromFrame(uniform_image, -10);
    EXPECT_TRUE(coreset.points.empty());
}

TEST_F(CoresetTest, MergeCoresetsBasic) {
    Coreset coreset_a = buildCoresetFromFrame(uniform_image, 50);
    Coreset coreset_b = buildCoresetFromFrame(checkerboard, 75);
    
    int merge_size = 100;
    Coreset merged = mergeCoresets(coreset_a, coreset_b, merge_size);
    
    EXPECT_EQ(merged.points.size(), merge_size);
}

TEST_F(CoresetTest, MergeCoresetsNoDownsampling) {
    Coreset coreset_a = buildCoresetFromFrame(tiny_image, 2);
    Coreset coreset_b = buildCoresetFromFrame(tiny_image, 2);
    
    int merge_size = 10; // Larger than combined size
    Coreset merged = mergeCoresets(coreset_a, coreset_b, merge_size);
    
    EXPECT_EQ(merged.points.size(), 4); // 2 + 2, no downsampling needed
}

TEST_F(CoresetTest, MergeCoresetsContentPreservation) {
    Coreset coreset_a = buildCoresetFromFrame(uniform_image, 20); // All purple
    Coreset coreset_b = buildCoresetFromFrame(tiny_image, 4);     // Multi-colored
    
    Coreset merged = mergeCoresets(coreset_a, coreset_b, 30); // No downsampling
    
    EXPECT_EQ(merged.points.size(), 24); // 20 + 4
    
    // Count purple points from uniform image
    int purple_count = 0;
    for (const auto& point : merged.points) {
        if (point.bgr[0] == 128.0f && point.bgr[1] == 64.0f && point.bgr[2] == 255.0f) {
            purple_count++;
        }
    }
    EXPECT_EQ(purple_count, 20);
}

TEST_F(CoresetTest, MergeCoresetsEmptyInputs) {
    Coreset empty_a, empty_b;
    Coreset merged = mergeCoresets(empty_a, empty_b, 100);
    EXPECT_TRUE(merged.points.empty());
}

TEST_F(CoresetTest, MergeCoresetsOneEmpty) {
    Coreset coreset_a = buildCoresetFromFrame(uniform_image, 50);
    Coreset empty_b;
    
    Coreset merged1 = mergeCoresets(coreset_a, empty_b, 100);
    EXPECT_EQ(merged1.points.size(), 50);
    
    Coreset merged2 = mergeCoresets(empty_b, coreset_a, 100);
    EXPECT_EQ(merged2.points.size(), 50);
}

TEST_F(CoresetTest, MergeCoresetsDownsamplingRandomness) {
    Coreset coreset_a = buildCoresetFromFrame(gradient_image, 100);
    Coreset coreset_b = buildCoresetFromFrame(checkerboard, 100);
    
    int small_size = 50;
    
    // Multiple merges with downsampling should differ
    Coreset merged1 = mergeCoresets(coreset_a, coreset_b, small_size);
    Coreset merged2 = mergeCoresets(coreset_a, coreset_b, small_size);
    
    EXPECT_EQ(merged1.points.size(), small_size);
    EXPECT_EQ(merged2.points.size(), small_size);
    
    bool different = false;
    for (size_t i = 0; i < merged1.points.size() && i < merged2.points.size(); ++i) {
        if (merged1.points[i].x != merged2.points[i].x || 
            merged1.points[i].y != merged2.points[i].y ||
            merged1.points[i].bgr[0] != merged2.points[i].bgr[0]) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

TEST_F(CoresetTest, MergeCoresetsLargeScale) {
    Coreset coreset_a = buildCoresetFromFrame(gradient_image, 1000);
    Coreset coreset_b = buildCoresetFromFrame(checkerboard, 1500);
    
    int merge_size = 800;
    Coreset merged = mergeCoresets(coreset_a, coreset_b, merge_size);
    
    EXPECT_EQ(merged.points.size(), merge_size);
    
    // Validate point properties
    for (const auto& point : merged.points) {
        EXPECT_GE(point.x, 0.0f);
        EXPECT_LE(point.x, 1.0f);
        EXPECT_GE(point.y, 0.0f);
        EXPECT_LE(point.y, 1.0f);
        EXPECT_GT(point.weight, 0.0f);
        EXPECT_GE(point.bgr[0], 0.0f);
        EXPECT_LE(point.bgr[0], 255.0f);
        EXPECT_GE(point.bgr[1], 0.0f);
        EXPECT_LE(point.bgr[1], 255.0f);
        EXPECT_GE(point.bgr[2], 0.0f);
        EXPECT_LE(point.bgr[2], 255.0f);
    }
}

TEST_F(CoresetTest, MergeCoresetsZeroSampleSize) {
    Coreset coreset_a = buildCoresetFromFrame(uniform_image, 50);
    Coreset coreset_b = buildCoresetFromFrame(checkerboard, 50);
    
    Coreset merged = mergeCoresets(coreset_a, coreset_b, 0);
    EXPECT_TRUE(merged.points.empty());
}

TEST_F(CoresetTest, BuildCoresetConsistentWeights) {
    std::vector<int> sample_sizes = {50, 100, 200, 500};
    int total_pixels = uniform_image.rows * uniform_image.cols;
    
    for (int sample_size : sample_sizes) {
        Coreset coreset = buildCoresetFromFrame(uniform_image, sample_size);
        float expected_weight = static_cast<float>(total_pixels) / static_cast<float>(sample_size);
        
        for (const auto& point : coreset.points) {
            EXPECT_FLOAT_EQ(point.weight, expected_weight);
        }
    }
}

TEST_F(CoresetTest, MergeCoresetsStressTest) {
    Coreset base = buildCoresetFromFrame(gradient_image, 100);
    
    // Iterative merging to test stability
    for (int i = 0; i < 10; ++i) {
        Coreset additional = buildCoresetFromFrame(checkerboard, 50);
        base = mergeCoresets(base, additional, 120);
        
        EXPECT_EQ(base.points.size(), 120);
        
        // Verify no NaN values
        for (const auto& point : base.points) {
            EXPECT_FALSE(std::isnan(point.x));
            EXPECT_FALSE(std::isnan(point.y));
            EXPECT_FALSE(std::isnan(point.weight));
            EXPECT_GT(point.weight, 0.0f);
        }
    }
}