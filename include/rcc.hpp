#pragma once
#include "coreset.hpp"
#include <vector>

// A node in the RCC tree, holding a coreset and pointers to left and right children
// If leaf, both children are nullptr
// - `coreset`: the coreset represented by this node
// - `left`, `right`: pointers to left and right child nodes (nullptr if leaf)
struct RCCNode {
	Coreset coreset;
	RCCNode* left = nullptr;
	RCCNode* right = nullptr;

	RCCNode(const Coreset& c) : coreset(c) {}
};

// The RCC (Recursive Cached Coreset Tree) structure for managing and merging coresets
class RCC {
private:
	RCCNode* root = nullptr;
	std::vector<RCCNode*> levels; // Each index represents a level in the tree; nullptr if empty
	int max_levels = 8; // Maximum number of levels in the tree. Helps to keep its height bounded

public:
	// Default constructor and destructor
	RCC() = default;
	~RCC();

	// Initialize RCC with a maximum number of levels
	// Args:
	//   maxLevels: maximum number of levels in the RCC tree
	//			  (higher means larger memory usage but potentially better accuracy)
	explicit RCC(int maxLevels) : max_levels(maxLevels) { levels.assign(std::max(1, max_levels), nullptr); }

	// Insert a new leaf coreset into the RCC tree, merging as necessary to maintain structure
	//
	// Args:
	//   leafCoreset: the new coreset to insert as a leaf
	//   sample_size: how many pixels to randomly sample for coreset
	//
    void insertLeaf(const Coreset& leafCoreset, int sample_size);

	// Merge two RCC nodes into one by merging their coresets and creating a new parent node
	// 
	// Args:
	//   nodeA, nodeB: the two RCC nodes to merge
	//   sample_size: how many pixels to randomly sample for coreset
	// 
	// Returns:
	//   A new RCCNode whose coreset is the merged coreset of A and B, with A and B as children
    RCCNode* mergeNodes(RCCNode* nodeA, RCCNode* nodeB, int sample_size);

	// Get the coreset at the root of the RCC tree
	// 
	// Returns:
	//   The root Coreset, or an empty Coreset if the tree is empty.
    Coreset getRootCoreset() const;
};