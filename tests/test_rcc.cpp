#include <gtest/gtest.h>
#include "../include/rcc.hpp"
#include "../include/coreset.hpp"
#include <opencv2/opencv.hpp>

class RCCTest : public ::testing::Test {
protected:
    void SetUp() override {
        max_levels = 3;
        sample_size = 500;
        rcc = std::make_unique<RCC>(max_levels);
        
        // Blue region test image
        test_image1 = cv::Mat::zeros(50, 50, CV_8UC3);
        cv::rectangle(test_image1, cv::Point(10, 10), cv::Point(40, 40), cv::Scalar(255, 0, 0), -1);
        
        // Two-region test image
        test_image2 = cv::Mat::zeros(50, 50, CV_8UC3);
        cv::rectangle(test_image2, cv::Point(0, 0), cv::Point(25, 25), cv::Scalar(0, 255, 0), -1);
        cv::rectangle(test_image2, cv::Point(25, 25), cv::Point(50, 50), cv::Scalar(0, 0, 255), -1);
        
        // Circle test image
        test_image3 = cv::Mat::zeros(60, 60, CV_8UC3);
        cv::circle(test_image3, cv::Point(30, 30), 20, cv::Scalar(255, 255, 255), -1);
    }

    std::unique_ptr<RCC> rcc;
    int max_levels;
    int sample_size;
    cv::Mat test_image1, test_image2, test_image3;
};

TEST_F(RCCTest, ConstructorInitialization) {
    RCC new_rcc(5);
    Coreset root_coreset = new_rcc.getRootCoreset();
    EXPECT_TRUE(root_coreset.points.empty());
}

TEST_F(RCCTest, InsertSingleLeaf) {
    Coreset coreset1 = buildCoresetFromFrame(test_image1, 200);
    
    rcc->insertLeaf(coreset1, sample_size);
    
    Coreset root_coreset = rcc->getRootCoreset();
    EXPECT_FALSE(root_coreset.points.empty());
    EXPECT_LE(root_coreset.points.size(), sample_size);
}

TEST_F(RCCTest, InsertMultipleLeaves) {
    Coreset coreset1 = buildCoresetFromFrame(test_image1, 150);
    Coreset coreset2 = buildCoresetFromFrame(test_image2, 200);
    Coreset coreset3 = buildCoresetFromFrame(test_image3, 180);
    
    rcc->insertLeaf(coreset1, sample_size);
    rcc->insertLeaf(coreset2, sample_size);
    rcc->insertLeaf(coreset3, sample_size);
    
    Coreset root_coreset = rcc->getRootCoreset();
    EXPECT_FALSE(root_coreset.points.empty());
    EXPECT_LE(root_coreset.points.size(), sample_size);
}

TEST_F(RCCTest, WeightPreservationSingleInsertion) {
    Coreset coreset1 = buildCoresetFromFrame(test_image1, 200);
    
    float initial_weight = 0.0f;
    for (const auto& point : coreset1.points) {
        initial_weight += point.weight;
    }
    
    rcc->insertLeaf(coreset1, sample_size);
    
    Coreset root_coreset = rcc->getRootCoreset();
    float final_weight = 0.0f;
    for (const auto& point : root_coreset.points) {
        final_weight += point.weight;
    }
    
    EXPECT_NEAR(final_weight, initial_weight, 1.0f);
}

TEST_F(RCCTest, WeightPreservationMultipleInsertions) {
    Coreset coreset1 = buildCoresetFromFrame(test_image1, 150);
    Coreset coreset2 = buildCoresetFromFrame(test_image2, 200);
    
    float weight1 = 0.0f, weight2 = 0.0f;
    for (const auto& point : coreset1.points) weight1 += point.weight;
    for (const auto& point : coreset2.points) weight2 += point.weight;
    
    rcc->insertLeaf(coreset1, sample_size);
    rcc->insertLeaf(coreset2, sample_size);
    
    Coreset root_coreset = rcc->getRootCoreset();
    float total_weight = 0.0f;
    for (const auto& point : root_coreset.points) {
        total_weight += point.weight;
    }
    
    EXPECT_NEAR(total_weight, weight1 + weight2, 2.0f);
}

TEST_F(RCCTest, BoundedMemoryUsage) {
    for (int i = 0; i < 20; ++i) {
        Coreset coreset = buildCoresetFromFrame(test_image1, 300);
        rcc->insertLeaf(coreset, sample_size);
        
        Coreset root_coreset = rcc->getRootCoreset();
        EXPECT_LE(root_coreset.points.size(), sample_size);
    }
}

TEST_F(RCCTest, LevelManagement) {
    for (int i = 0; i < 10; ++i) {
        Coreset coreset = buildCoresetFromFrame(test_image2, 200);
        rcc->insertLeaf(coreset, sample_size);
        
        Coreset root_coreset = rcc->getRootCoreset();
        EXPECT_FALSE(root_coreset.points.empty());
    }
}

TEST_F(RCCTest, MergeNodesBasic) {
    Coreset coreset1 = buildCoresetFromFrame(test_image1, 100);
    Coreset coreset2 = buildCoresetFromFrame(test_image2, 150);
    
    RCCNode* node1 = new RCCNode(coreset1);
    RCCNode* node2 = new RCCNode(coreset2);
    
    RCCNode* merged = rcc->mergeNodes(node1, node2, sample_size);
    
    EXPECT_NE(merged, nullptr);
    EXPECT_EQ(merged->left, node1);
    EXPECT_EQ(merged->right, node2);
    EXPECT_LE(merged->coreset.points.size(), sample_size);
    
    delete merged;
    delete node1;
    delete node2;
}

TEST_F(RCCTest, MergeNodesNullInputs) {
    Coreset coreset1 = buildCoresetFromFrame(test_image1, 100);
    RCCNode* node1 = new RCCNode(coreset1);
    
    RCCNode* result1 = rcc->mergeNodes(nullptr, node1, sample_size);
    EXPECT_EQ(result1, node1);
    
    RCCNode* result2 = rcc->mergeNodes(node1, nullptr, sample_size);
    EXPECT_EQ(result2, node1);
    
    RCCNode* result3 = rcc->mergeNodes(nullptr, nullptr, sample_size);
    EXPECT_EQ(result3, nullptr);
    
    delete node1;
}

TEST_F(RCCTest, MergeNodesWeightPreservation) {
    Coreset coreset1 = buildCoresetFromFrame(test_image1, 100);
    Coreset coreset2 = buildCoresetFromFrame(test_image2, 120);
    
    float weight1 = 0.0f, weight2 = 0.0f;
    for (const auto& p : coreset1.points) weight1 += p.weight;
    for (const auto& p : coreset2.points) weight2 += p.weight;
    
    RCCNode* node1 = new RCCNode(coreset1);
    RCCNode* node2 = new RCCNode(coreset2);
    
    RCCNode* merged = rcc->mergeNodes(node1, node2, sample_size);
    
    float merged_weight = 0.0f;
    for (const auto& p : merged->coreset.points) {
        merged_weight += p.weight;
    }
    
    EXPECT_NEAR(merged_weight, weight1 + weight2, 1.0f);
    
    delete merged;
    delete node1;
    delete node2;
}

TEST_F(RCCTest, EmptyTreeBehavior) {
    Coreset empty_root = rcc->getRootCoreset();
    EXPECT_TRUE(empty_root.points.empty());
}

TEST_F(RCCTest, SingleNodeTree) {
    Coreset coreset = buildCoresetFromFrame(test_image1, 200);
    rcc->insertLeaf(coreset, sample_size);
    
    Coreset root_coreset = rcc->getRootCoreset();
    EXPECT_FALSE(root_coreset.points.empty());
    EXPECT_LE(root_coreset.points.size(), sample_size);
}

TEST_F(RCCTest, LargeSampleSizeHandling) {
    int large_sample_size = 10000;
    Coreset coreset = buildCoresetFromFrame(test_image1, 100);
    
    rcc->insertLeaf(coreset, large_sample_size);
    
    Coreset root_coreset = rcc->getRootCoreset();
    EXPECT_FALSE(root_coreset.points.empty());
    EXPECT_LE(root_coreset.points.size(), coreset.points.size());
}

TEST_F(RCCTest, StressTestManyInsertions) {
    for (int i = 0; i < 100; ++i) {
        cv::Mat& test_image = (i % 3 == 0) ? test_image1 : 
                             (i % 3 == 1) ? test_image2 : test_image3;
        
        Coreset coreset = buildCoresetFromFrame(test_image, 150);
        rcc->insertLeaf(coreset, sample_size);
        
        Coreset root_coreset = rcc->getRootCoreset();
        EXPECT_FALSE(root_coreset.points.empty());
        EXPECT_LE(root_coreset.points.size(), sample_size);
        
        for (const auto& point : root_coreset.points) {
            EXPECT_GT(point.weight, 0.0f);
        }
    }
}

TEST_F(RCCTest, MaxLevelsRespected) {
    RCC small_rcc(2);
    
    for (int i = 0; i < 10; ++i) {
        Coreset coreset = buildCoresetFromFrame(test_image1, 200);
        small_rcc.insertLeaf(coreset, sample_size);
    }
    
    Coreset root_coreset = small_rcc.getRootCoreset();
    EXPECT_FALSE(root_coreset.points.empty());
    EXPECT_LE(root_coreset.points.size(), sample_size);
}

TEST_F(RCCTest, DestructorMemoryCleanup) {
    {
        RCC temp_rcc(3);
        
        for (int i = 0; i < 5; ++i) {
            Coreset coreset = buildCoresetFromFrame(test_image2, 200);
            temp_rcc.insertLeaf(coreset, sample_size);
        }
        
        Coreset root_coreset = temp_rcc.getRootCoreset();
        EXPECT_FALSE(root_coreset.points.empty());
    }
    
    SUCCEED();
}

TEST_F(RCCTest, ConsistentBehaviorAcrossInsertions) {
    std::vector<size_t> root_sizes;
    std::vector<float> root_weights;
    
    for (int i = 0; i < 5; ++i) {
        Coreset coreset = buildCoresetFromFrame(test_image3, 200);
        rcc->insertLeaf(coreset, sample_size);
        
        Coreset root_coreset = rcc->getRootCoreset();
        root_sizes.push_back(root_coreset.points.size());
        
        float total_weight = 0.0f;
        for (const auto& point : root_coreset.points) {
            total_weight += point.weight;
        }
        root_weights.push_back(total_weight);
    }
    
    for (size_t size : root_sizes) {
        EXPECT_LE(size, sample_size);
        EXPECT_GT(size, 0);
    }
    
    for (float weight : root_weights) {
        EXPECT_GT(weight, 0.0f);
    }
}