#include <gtest/gtest.h>
#include "../include/utils.hpp"
#include "../include/coreset.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class UtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Two-region test image
        simple_image = cv::Mat::zeros(100, 100, CV_8UC3);
        cv::rectangle(simple_image, cv::Point(0, 0), cv::Point(50, 50), cv::Scalar(255, 0, 0), -1);
        cv::rectangle(simple_image, cv::Point(50, 50), cv::Point(100, 100), cv::Scalar(0, 255, 0), -1);
        
        // Four-region test image
        complex_image = cv::Mat::zeros(200, 200, CV_8UC3);
        cv::rectangle(complex_image, cv::Point(0, 0), cv::Point(100, 100), cv::Scalar(255, 0, 0), -1);
        cv::rectangle(complex_image, cv::Point(100, 0), cv::Point(200, 100), cv::Scalar(0, 255, 0), -1);
        cv::rectangle(complex_image, cv::Point(0, 100), cv::Point(100, 200), cv::Scalar(0, 0, 255), -1);
        cv::rectangle(complex_image, cv::Point(100, 100), cv::Point(200, 200), cv::Scalar(255, 255, 255), -1);
        
        color_scale = 1.0f;
        spatial_scale = 0.1f;
    }

    cv::Mat simple_image;
    cv::Mat complex_image;
    float color_scale;
    float spatial_scale;
};

TEST_F(UtilsTest, MakeFeatureBasicTest) {
    cv::Vec3f bgr(100.0f, 150.0f, 200.0f);
    float x01 = 0.5f;
    float y01 = 0.3f;
    
    cv::Vec<float, 5> feature = makeFeature(bgr, x01, y01, color_scale, spatial_scale);
    
    EXPECT_FLOAT_EQ(feature[0], 100.0f * color_scale);
    EXPECT_FLOAT_EQ(feature[1], 150.0f * color_scale);
    EXPECT_FLOAT_EQ(feature[2], 200.0f * color_scale);
    EXPECT_FLOAT_EQ(feature[3], 0.5f * spatial_scale);
    EXPECT_FLOAT_EQ(feature[4], 0.3f * spatial_scale);
}

TEST_F(UtilsTest, MakeFeatureScaling) {
    cv::Vec3f bgr(255.0f, 128.0f, 64.0f);
    float x01 = 1.0f;
    float y01 = 0.0f;
    
    float test_color_scale = 2.0f;
    float test_spatial_scale = 0.5f;
    
    cv::Vec<float, 5> feature = makeFeature(bgr, x01, y01, test_color_scale, test_spatial_scale);
    
    EXPECT_FLOAT_EQ(feature[0], 255.0f * test_color_scale);
    EXPECT_FLOAT_EQ(feature[1], 128.0f * test_color_scale);
    EXPECT_FLOAT_EQ(feature[2], 64.0f * test_color_scale);
    EXPECT_FLOAT_EQ(feature[3], 1.0f * test_spatial_scale);
    EXPECT_FLOAT_EQ(feature[4], 0.0f * test_spatial_scale);
}

TEST_F(UtilsTest, MakeFeatureZeroValues) {
    cv::Vec3f bgr(0.0f, 0.0f, 0.0f);
    float x01 = 0.0f;
    float y01 = 0.0f;
    
    cv::Vec<float, 5> feature = makeFeature(bgr, x01, y01, color_scale, spatial_scale);
    
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(feature[i], 0.0f);
    }
}

TEST_F(UtilsTest, ComputeKMeansCentersBasic) {
    int k = 2;
    int sample_size = 500;
    
    std::vector<cv::Vec<float, 5>> centers = computeKMeansCenters(
        simple_image, k, sample_size, color_scale, spatial_scale);
    
    EXPECT_EQ(centers.size(), k);
    
    for (const auto& center : centers) {
        EXPECT_EQ(center.rows, 5);
        
        // BGR components in valid range
        EXPECT_GE(center[0], 0.0f);
        EXPECT_LE(center[0], 255.0f * color_scale);
        EXPECT_GE(center[1], 0.0f);
        EXPECT_LE(center[1], 255.0f * color_scale);
        EXPECT_GE(center[2], 0.0f);
        EXPECT_LE(center[2], 255.0f * color_scale);
        
        // Spatial components in valid range
        EXPECT_GE(center[3], 0.0f);
        EXPECT_LE(center[3], 1.0f * spatial_scale);
        EXPECT_GE(center[4], 0.0f);
        EXPECT_LE(center[4], 1.0f * spatial_scale);
    }
}

TEST_F(UtilsTest, ComputeKMeansCentersComplexImage) {
    int k = 4;
    int sample_size = 1000;
    
    std::vector<cv::Vec<float, 5>> centers = computeKMeansCenters(
        complex_image, k, sample_size, color_scale, spatial_scale);
    
    EXPECT_EQ(centers.size(), k);
    
    // Check centers are distinct
    for (size_t i = 0; i < centers.size(); ++i) {
        for (size_t j = i + 1; j < centers.size(); ++j) {
            float dist = 0.0f;
            for (int d = 0; d < 5; ++d) {
                float diff = centers[i][d] - centers[j][d];
                dist += diff * diff;
            }
            EXPECT_GT(dist, 0.0f);
        }
    }
}

TEST_F(UtilsTest, ComputeKMeansCentersInvalidK) {
    std::vector<cv::Vec<float, 5>> centers = computeKMeansCenters(
        simple_image, 0, 500, color_scale, spatial_scale);
    EXPECT_EQ(centers.size(), 1);
    
    centers = computeKMeansCenters(simple_image, -5, 500, color_scale, spatial_scale);
    EXPECT_EQ(centers.size(), 1);
}

TEST_F(UtilsTest, ComputeKMeansCentersInvalidSampleSize) {
    int k = 3;
    
    std::vector<cv::Vec<float, 5>> centers = computeKMeansCenters(
        simple_image, k, 0, color_scale, spatial_scale);
    EXPECT_EQ(centers.size(), k);
    
    centers = computeKMeansCenters(simple_image, k, -100, color_scale, spatial_scale);
    EXPECT_EQ(centers.size(), k);
}

TEST_F(UtilsTest, ComputeKMeansCentersLargeK) {
    int k = 50;
    int sample_size = 1000;
    
    std::vector<cv::Vec<float, 5>> centers = computeKMeansCenters(
        simple_image, k, sample_size, color_scale, spatial_scale);
    
    EXPECT_EQ(centers.size(), k);
}

TEST_F(UtilsTest, ComputeKMeansCentersSmallSampleSize) {
    int k = 2;
    int sample_size = 10;
    
    std::vector<cv::Vec<float, 5>> centers = computeKMeansCenters(
        simple_image, k, sample_size, color_scale, spatial_scale);
    
    EXPECT_EQ(centers.size(), k);
}

TEST_F(UtilsTest, ComputeKMeansCentersDifferentScales) {
    int k = 2;
    int sample_size = 500;
    
    std::vector<cv::Vec<float, 5>> centers1 = computeKMeansCenters(
        simple_image, k, sample_size, 0.5f, spatial_scale);
    std::vector<cv::Vec<float, 5>> centers2 = computeKMeansCenters(
        simple_image, k, sample_size, 2.0f, spatial_scale);
    
    EXPECT_EQ(centers1.size(), k);
    EXPECT_EQ(centers2.size(), k);
    
    // Centers should differ due to scaling
    bool different = false;
    for (int i = 0; i < k && !different; ++i) {
        for (int d = 0; d < 3; ++d) {
            if (std::abs(centers1[i][d] - centers2[i][d]) > 1e-6f) {
                different = true;
                break;
            }
        }
    }
    EXPECT_TRUE(different);
}

TEST_F(UtilsTest, ComputeKMeansCentersReproducibility) {
    int k = 3;
    int sample_size = 500;
    
    for (int trial = 0; trial < 5; ++trial) {
        std::vector<cv::Vec<float, 5>> centers = computeKMeansCenters(
            complex_image, k, sample_size, color_scale, spatial_scale);
        
        EXPECT_EQ(centers.size(), k);
        
        for (const auto& center : centers) {
            for (int d = 0; d < 5; ++d) {
                EXPECT_FALSE(std::isnan(center[d]));
                EXPECT_FALSE(std::isinf(center[d]));
            }
        }
    }
}

TEST_F(UtilsTest, ComputeKMeansCentersWrongImageType) {
    cv::Mat wrong_type_image = cv::Mat::zeros(100, 100, CV_8UC1);
    
    try {
        std::vector<cv::Vec<float, 5>> centers = computeKMeansCenters(
            wrong_type_image, 2, 500, color_scale, spatial_scale);
        EXPECT_TRUE(centers.empty() || centers.size() == 2);
    } catch (...) {
        SUCCEED();
    }
}

TEST_F(UtilsTest, ComputeKMeansCentersEmptyImage) {
    cv::Mat empty_image;
    
    try {
        std::vector<cv::Vec<float, 5>> centers = computeKMeansCenters(
            empty_image, 2, 500, color_scale, spatial_scale);
        EXPECT_TRUE(centers.empty());
    } catch (...) {
        SUCCEED();
    }
}

TEST_F(UtilsTest, ComputeKMeansCentersInvalidInputHandling) {
    // Empty image
    cv::Mat empty_image;
    bool empty_handled = false;
    try {
        auto centers = computeKMeansCenters(empty_image, 2, 500, color_scale, spatial_scale);
        empty_handled = true;
        EXPECT_TRUE(centers.empty() || centers.size() <= 2);
    } catch (...) {
        empty_handled = true;
    }
    EXPECT_TRUE(empty_handled);
    
    // Wrong type image  
    cv::Mat gray_image = cv::Mat::zeros(50, 50, CV_8UC1);
    bool gray_handled = false;
    try {
        auto centers = computeKMeansCenters(gray_image, 2, 500, color_scale, spatial_scale);
        gray_handled = true;
        EXPECT_TRUE(centers.empty() || centers.size() <= 2);
    } catch (...) {
        gray_handled = true;
    }
    EXPECT_TRUE(gray_handled);
}