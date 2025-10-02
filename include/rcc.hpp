struct RCCNode {
	Coreset coreset;
	RCCNode* left;
	RCCNode* right;
};

Coreset getRootCoreset();

void mergeUp();

void insertLeaf(Coreset leaf);