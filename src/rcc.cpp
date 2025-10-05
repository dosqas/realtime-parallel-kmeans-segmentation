#include "rcc.hpp"
#include "coreset.hpp"
#include <unordered_set>

// Delete a subtree of our RCC. Used in cleaning up when deleting and pruning the RCC
static void deleteSubtree(RCCNode* node) 
{
	if (!node) return;              // Base case - RCCNode is null, meaning we reached the end of the tree
    deleteSubtree(node->left);      // Recursively call the delete function on the left
    deleteSubtree(node->right);     // and right subtrees
    delete node;                    // Once we return from the recursion, we delete the node
}

// Insert a leaf in the RCC tree, merging as necessary to maintain structure
void RCC::insertLeaf(const Coreset& leafCoreset, int sample_size) 
{
	RCCNode* carry = new RCCNode(leafCoreset); // New leaf node to insert

    if (levels.empty()) {
        levels.assign(std::max(1, this->max_levels), nullptr);
    }

    for (int lvl = 0; lvl < (int)levels.size(); ++lvl) {
		// If level is empty, place carry here and stop
        if (!levels[lvl]) {
            levels[lvl] = carry;
            carry = nullptr;
            break;
        } else {
			carry = mergeNodes(levels[lvl], carry, sample_size); // Merge carry with existing node at this level
            levels[lvl] = nullptr;
        }
    }

    // If still have a carry beyond max_levels, drop oldest by merging into last level
    if (carry) {
        // bounded cap: merge into top level and replace
        if (levels.back()) {
			carry = mergeNodes(levels.back(), carry, sample_size); // Merge with existing top level
			deleteSubtree(levels.back()); // Delete the old top level subtree to avoid memory leak
        }
        levels.back() = carry;
    }

    // Recompute root as merge of all non-null levels (low to high)
    RCCNode* newRoot = nullptr;
    for (int lvl = 0; lvl < (int)levels.size(); ++lvl) {
        if (!levels[lvl]) continue;

        // Merge into new root if not null, else set as root
		newRoot = newRoot ? mergeNodes(newRoot, levels[lvl], sample_size) : levels[lvl];
    }
    root = newRoot;
}

// Merge two RCC nodes A and B into one by merging their coresets and creating a new parent node
// having A and B as children
RCCNode* RCC::mergeNodes(RCCNode* nodeA, RCCNode* nodeB, int sample_size) 
{
    if (!nodeA) return nodeB;
    if (!nodeB) return nodeA;

	Coreset merged = mergeCoresets(nodeA->coreset, nodeB->coreset, sample_size); // Merge the coresets of A and B
	RCCNode* parent = new RCCNode(merged); // Create a new parent node with the merged coreset
    parent->left = nodeA;
    parent->right = nodeB;

    return parent;
}

// Get the coreset at the root of the RCC tree
Coreset RCC::getRootCoreset() const 
{
	if (!root) return Coreset{}; // Return empty coreset if tree is empty

    return root->coreset;
}

RCC::~RCC() 
{
	// Delete unique subtrees from the level array to avoid double-free errors.
    // Track deletions by marking pointers we already removed
    std::unordered_set<RCCNode*> seen;
    for (RCCNode* node : levels) {
        if (node && !seen.count(node)) {
            deleteSubtree(node);
            seen.insert(node);
        }
    }

    // In case levels wasn't used, ensure root is deleted.
    if (!levels.size() && root) {
        deleteSubtree(root);
    }
}
