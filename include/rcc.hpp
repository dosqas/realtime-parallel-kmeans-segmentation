#pragma once
#include "coreset.hpp"
#include <vector>

struct RCCNode {
	Coreset coreset;
	RCCNode* left = nullptr;
	RCCNode* right = nullptr;

	RCCNode(const Coreset& c) : coreset(c) {}
};

class RCC {
private:
	RCCNode* root = nullptr;
	std::vector<RCCNode*> levels;
	int max_levels = 8;

public:
	RCC() = default;
	explicit RCC(int maxLevels) : max_levels(maxLevels) { levels.assign(std::max(1, max_levels), nullptr); }
	~RCC();

    void insertLeaf(const Coreset& leafCoreset, int sample_size);

    RCCNode* mergeNodes(RCCNode* nodeA, RCCNode* nodeB, int sample_size);

    Coreset getRootCoreset() const;
};