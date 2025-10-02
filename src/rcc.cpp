#include "rcc.hpp"
#include "coreset.hpp"
#include <unordered_set>

static void deleteSubtree(RCCNode* node) {
    if (!node) return;
    deleteSubtree(node->left);
    deleteSubtree(node->right);
    delete node;
}

void RCC::insertLeaf(const Coreset& leafCoreset, int sample_size) {
    RCCNode* carry = new RCCNode(leafCoreset);

    if (levels.empty()) {
        levels.assign(std::max(1, this->max_levels), nullptr);
    }

    for (int lvl = 0; lvl < (int)levels.size(); ++lvl) {
        if (!levels[lvl]) {
            levels[lvl] = carry;
            carry = nullptr;
            break;
        } else {
            carry = mergeNodes(levels[lvl], carry, sample_size);
            levels[lvl] = nullptr;
        }
    }

    // If still have a carry beyond max_levels, drop oldest by merging into last level
    if (carry) {
        // bounded cap: merge into top level and replace
        if (levels.back()) {
            carry = mergeNodes(levels.back(), carry, sample_size);
            deleteSubtree(levels.back());
        }
        levels.back() = carry;
    }

    // Recompute root as merge of all non-null levels (low to high)
    RCCNode* newRoot = nullptr;
    for (int lvl = 0; lvl < (int)levels.size(); ++lvl) {
        if (!levels[lvl]) continue;
        newRoot = newRoot ? mergeNodes(newRoot, levels[lvl], sample_size) : levels[lvl];
    }
    root = newRoot;
}

RCCNode* RCC::mergeNodes(RCCNode* nodeA, RCCNode* nodeB, int sample_size) {
    if (!nodeA) return nodeB;
    if (!nodeB) return nodeA;
    Coreset merged = mergeCoresets(nodeA->coreset, nodeB->coreset, sample_size);
    RCCNode* parent = new RCCNode(merged);
    parent->left = nodeA;
    parent->right = nodeB;
    return parent;
}

Coreset RCC::getRootCoreset() const {
    if (!root) return Coreset{};
    return root->coreset;
}

RCC::~RCC() {
    // Delete unique subtrees from the level array to avoid double-free
    // Track deletions by marking pointers we already removed.
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
