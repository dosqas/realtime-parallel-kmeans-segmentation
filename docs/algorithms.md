# Algorithms

## Overview

This project implements real-time image segmentation using K-Means clustering, with several parallel and one sequential backends. The core algorithm is K-Means, adapted for image segmentation by combining color and spatial features. The system uses **OpenCV** for image processing and video I/O, and employs a **Recursive Cached Coreset (RCC) Tree** structure for efficient real-time streaming segmentation. The following describes the main algorithmic steps and design choices.

---

## K-Means Clustering for Image Segmentation

### Feature Representation

Each pixel is represented as a 5-dimensional feature vector:
- **[B, G, R, x, y]**
  - B, G, R: Blue, Green, Red color channels (scaled by `color_scale`)
  - x, y: Normalized spatial coordinates (scaled by `spatial_scale`)

This allows the algorithm to cluster pixels based on both color similarity and spatial proximity, producing spatially coherent segments.

### Coreset Sampling

To accelerate clustering, a random subset (`sample_size`) of pixels is selected as a coreset. K-Means is run on this subset to find initial cluster centers, reducing computational cost while maintaining segmentation quality.

Each coreset point contains:
- **BGR color**: Average color of represented pixels
- **Spatial coordinates**: Normalized (x, y) position in [0, 1] range  
- **Weight**: Number of original pixels this point represents

### K-Means Steps

1. **Initialization**
   - Randomly select `k` initial cluster centers from the coreset.

2. **Assignment Step**
   - For each pixel, compute the squared Euclidean distance in 5D feature space to each cluster center.
   - Assign the pixel to the nearest cluster.

3. **Update Step**
   - For each cluster, recompute the center as the mean of all assigned pixel features.

4. **Repeat**
   - Iterate assignment and update steps until convergence or a maximum number of iterations is reached.

5. **Segmentation Output**
   - Each pixel is recolored to the color of its assigned cluster center (inverse-scaled to original color range).

---

## Recursive Cached Coreset (RCC) Tree

### Overview

The project implements a **Recursive Cached Coreset (RCC) Tree** structure, which is particularly well-suited for real-time streaming applications that perform K-means segmentation. The RCC tree efficiently manages and merges coresets over time, maintaining bounded memory usage while preserving clustering quality.

### RCC Tree Structure

The RCC tree is a binary tree where:
- **Nodes**: Each node contains a coreset representing a compressed subset of pixels
- **Leaves**: New coresets from individual frames are inserted as leaves
- **Internal Nodes**: Created by merging child coresets, with pointers to left and right children
- **Levels**: The tree maintains a level-based structure with bounded height (default: 8 levels)

### Tree Operations

#### 1. Leaf Insertion
When a new frame coreset arrives:
1. Create a new leaf node with the frame's coreset
2. **Carry propagation**: Starting from level 0, if a level is empty, place the new node there
3. If a level is occupied, merge the new node with the existing node and propagate to the next level
4. **Bounded capacity**: If the tree exceeds maximum levels, merge into the top level to maintain memory bounds

#### 2. Node Merging
Two RCC nodes are merged by:
1. Combining their coresets using weighted sampling
2. Creating a new parent node with the merged coreset
3. Setting the original nodes as left and right children
4. If the merged coreset exceeds `sample_size`, randomly downsample to maintain bounds

#### 3. Root Computation
The root coreset is dynamically computed by merging all non-empty levels from bottom to top, providing a comprehensive representation of the entire stream history.

### Efficiency Benefits

- **Memory Bounded**: Tree height is limited, preventing unbounded growth
- **Temporal Decay**: Older coresets naturally get merged deeper in the tree
- **Fast Merging**: Coreset merging is much faster than re-clustering entire pixel sets
- **Quality Preservation**: Theoretical guarantees ensure clustering quality remains within (1±ε) of the full dataset

---

## Video I/O and Real-time Processing

### OpenCV Integration

The system uses **OpenCV** extensively for:
- **Video Capture**: Real-time webcam feed acquisition using `cv::VideoCapture`
- **Image Processing**: Frame manipulation, resizing, and color space operations
- **Display**: Live visualization with side-by-side original and segmented frames
- **User Interface**: Interactive sliders for K-value adjustment and keyboard controls for backend switching

### Real-time Pipeline

1. **Frame Acquisition**: Capture frames from webcam at 640x480 resolution
2. **Coreset Generation**: Build coreset from current frame using random sampling
3. **RCC Integration**: Insert frame coreset into RCC tree (for streaming scenarios)
4. **Segmentation**: Apply K-means clustering using selected backend
5. **Visualization**: Display original and segmented frames with performance metrics
6. **Interactive Controls**: 
   - Slider for K-value (1-12 clusters)
   - Keyboard shortcuts: '1'=Sequential, '2'=MPI, '3'=Threaded, '4'=CUDA

### Performance Monitoring

The system tracks and displays:
- **Current FPS**: Instantaneous frame rate
- **Min/Max FPS**: Performance range over last 3 seconds  
- **Backend switching**: Real-time comparison between implementations

---

## Backend-Specific Implementations

### Sequential (CPU)

- All steps are performed in a single thread.
- Used as a baseline for performance and correctness.
- Processes the entire image sequentially using standard K-means implementation.

### Multi-threaded (std::thread)

- Uses **standard C++ threads** (`std::thread`) for parallelization.
- **Work distribution**: Image is divided by rows, with each thread processing a contiguous chunk of rows.
- **Thread count**: Automatically determined using `std::thread::hardware_concurrency()` (typically number of CPU cores).
- **Synchronization**: Each thread processes distinct row ranges independently, then all threads are joined.
- **Memory safety**: Threads share read-only access to input frame and cluster centers, with exclusive write access to their assigned output rows.

### Distributed (MPI)

- The image is divided into blocks of rows, distributed across MPI processes.
- **Rank 0**: Computes cluster centers and broadcasts them to all ranks.
- **Worker ranks**: Each process assigns its rows' pixels to clusters independently.
- **Hybrid parallelization**: Uses OpenMP (`#pragma omp parallel for`) within each MPI rank for additional multi-threading.
- **Results gathering**: All segmented blocks are gathered at rank 0 using `MPI_Gatherv` for final output.
- Enables processing of very large images across multiple machines.

### GPU-accelerated (CUDA)

- **CPU preprocessing**: Cluster centers are computed on the CPU using the coreset.
- **GPU transfer**: Input image and cluster centers are transferred to GPU memory.
- **Parallel assignment**: CUDA kernel assigns pixels to clusters in parallel (one thread per pixel).
- **Thread organization**: Uses CUDA blocks and threads with linear indexing for pixel processing.
- **GPU transfer back**: Segmented image is copied back to CPU for display.
- Achieves significant speedup for large images with high parallelism.

---

## Mathematical Formulation

Given a set of pixel features $X = \{x_1, x_2, ..., x_n\}$, the goal is to find $k$ centers $C = \{c_1, ..., c_k\}$ that minimize:

$$
\sum_{i=1}^n \min_{j=1..k} \|x_i - c_j\|^2
$$

where $x_i$ is the 5D feature vector for pixel $i$.

### Coreset Theory

A coreset $S$ with weights $w_i$ approximates the original dataset such that:

$$
\left| \sum_{x \in X} \|x - c\|^2 - \sum_{s \in S} w_s \|s - c\|^2 \right| \leq \epsilon \cdot \text{OPT}
$$

where $\epsilon$ is the approximation error and $\text{OPT}$ is the optimal K-means cost.

### RCC Tree Merging

When merging coresets $A$ and $B$, the combined weight of each point is preserved:
- **Original weights**: Each point maintains its weight from source coreset
- **Random sampling**: If $|A| + |B| > \text{sample\_size}$, points are randomly selected
- **Weight preservation**: Total weight equals sum of original dataset sizes

---

## Implementation Details

### OpenCV Data Structures

- **cv::Mat**: Primary image container (3-channel BGR format)
- **cv::Vec3b**: 8-bit 3-channel pixel representation (0-255 range)
- **cv::Vec3f**: 32-bit floating-point color representation
- **cv::VideoCapture**: Real-time camera interface with configurable resolution

### Memory Management

- **Coreset Points**: Efficient representation with 5 floats per point (BGR + xy + weight)
- **RCC Tree Cleanup**: Recursive deletion prevents memory leaks in tree structure
- **Frame Processing**: In-place operations where possible to minimize memory allocation

### Sampling Strategy

- **Uniform Random Sampling**: Each pixel has equal probability of selection
- **Mersenne Twister PRNG**: High-quality random number generation for reproducible results
- **Weight Calculation**: `weight = total_pixels / sample_size` ensures proper representation

---

## Design Choices

- **Color and Spatial Scaling:**  
  The `color_scale` and `spatial_scale` parameters control the influence of color and position in clustering, allowing for tunable spatial coherence.
- **Coreset Sampling:**  
  Reduces computational cost for large images by clustering on a representative subset.
- **RCC Tree Structure:**  
  Enables efficient streaming processing by maintaining bounded memory usage while preserving historical information.
- **Multiple Backends:**  
  Allow leveraging multi-core CPUs, distributed systems, and GPUs for real-time performance.
- **OpenCV Integration:**  
  Provides robust image processing capabilities and cross-platform video I/O support.

---

## References

- K-Means Clustering: https://en.wikipedia.org/wiki/K-means_clustering
- Coreset Sampling: https://en.wikipedia.org/wiki/Coreset
- Recursive Cached Coreset Trees: Literature on efficient streaming K-means for real-time applications
- Image Segmentation with K-Means: [OpenCV Documentation](https://docs.opencv.org/master/d3/dc1/tutorial_kmeans_clustering.html)
- OpenCV Library: https://opencv.org/
- CUDA Programming: https://developer.nvidia.com/cuda-toolkit
- MPI Standard: https://www.mpi-forum.org/
- OpenMP API: https://www.openmp.org/
