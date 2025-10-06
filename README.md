# 🎨 Real-Time Parallel K-Means Image Segmentation

A high-performance computer vision system that performs real-time image segmentation using K-Means clustering with multiple parallel backends for optimal performance across different hardware configurations.

---

## ✨ Key Features

- **🚀 Real-Time Performance**: Up to 55+ FPS with CUDA backend on live webcam feeds
- **🔧 Multiple Parallel Backends**: Sequential, Multi-threaded, MPI, and CUDA implementations
- **🌳 RCC Tree Optimization**: Recursive Cached Coreset tree for efficient streaming segmentation
- **🎯 5D Feature Space**: Combines color (BGR) and spatial (x,y) features for coherent segmentation
- **⚡ Dynamic Backend Switching**: Switch between backends in real-time with keyboard shortcuts
- **📊 Performance Monitoring**: Live FPS tracking with min/max statistics
- **🖼️ Interactive Controls**: Adjustable K-value slider and side-by-side visualization

---

## 🧠 Technical Overview

This project implements an advanced K-Means clustering system optimized for real-time image segmentation with multiple parallelization strategies:

- **Core Algorithm**: K-Means clustering adapted for image segmentation
- **Feature Engineering**: 5D vectors combining color similarity and spatial proximity
- **Coreset Sampling**: Reduces computational complexity from O(n·k·t) to O(s·k·t), where n = total pixels, s = coreset size (s ≪ n)
- **RCC Tree Structure**: Maintains $(1 \pm \epsilon)$-approximation with bounded memory
- **Hardware Optimization**: Leverages multi-core CPUs, distributed systems, and GPUs

---

## 📊 Performance Benchmarks

### FPS Performance by Backend and K-value

| Backend | K=2 (Min FPS) | K=2 (Max FPS) | K=12 (Min FPS) | K=12 (Max FPS) | Performance Ratio |
|---------|---------------|---------------|----------------|----------------|-------------------|
| **Sequential** | 15 | 17 | 5 | 6 | 1.0× (baseline) |
| **MPI** | 13 | 31 | 10 | 13 | 1.8× average |
| **Multi-threaded** | 14 | 44 | 10 | 22 | 2.4× average |
| **CUDA** | 14 | 55 | 15 | 44 | **3.2× average** |

### Performance Characteristics

```
Performance Improvement Factor (vs Sequential):
┌─────────────────────────────────────────────────┐
│ CUDA:          ████████████████ 3.2×           │
│ Multi-thread:  ████████████ 2.4×               │
│ MPI:           ███████ 1.8×                     │
│ Sequential:    ████ 1.0× (baseline)            │
└─────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture Details

### Algorithmic Complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| **Sequential K-Means** | $O(n \cdot k \cdot t)$ | Baseline implementation |
| **Coreset K-Means** | $O(s \cdot k \cdot t)$ | Reduced complexity with $s \ll n$ |
| **RCC Tree Insertion** | $O(s \cdot \log N)$ | Streaming update per frame |
| **RCC Tree Merging** | $O(s)$ | Weighted merge operation |
| **GPU Assignment** | $O(n / \text{cores})$ | Massive parallelization |

### Backend-Specific Implementations

#### 🔄 Multi-threaded (std::thread)
- **Strategy**: Row-based work distribution across CPU threads
- **Synchronization**: Lock-free design with barrier synchronization
- **Memory Safety**: Read-only sharing with exclusive write regions
- **Best For**: Multi-core desktop systems

#### 🌐 Distributed (MPI + OpenMP)
- **Strategy**: Process-level distribution with thread-level parallelization
- **Communication**: MPI_Bcast for centers, MPI_Gatherv for results
- **Hybrid Approach**: Combines inter-process and intra-process parallelism
- **Best For**: HPC clusters and large distributed systems

#### 🎮 GPU-accelerated (CUDA)
- **Strategy**: One thread per pixel for maximum throughput
- **Memory Management**: Efficient host-device transfers
- **Synchronization**: cudaDeviceSynchronize() for completion barriers
- **Best For**: High-throughput applications with GPU hardware

---

## 🚀 Getting Started

### Prerequisites

```bash
# System Requirements
- C++17 compatible compiler (GCC/MSVC/Clang)
- CMake 3.18+
- OpenCV 4.0+
- CUDA Toolkit (for GPU backend)
- MPI implementation (for distributed backend)
```

### Building the Project

```bash
# Clone the repository
git clone https://github.com/dosqas/realtime-parallel-kmeans-segmentation.git
cd realtime-parallel-kmeans-segmentation

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build . --config Release

# Run the application
./realtime_segmentation  # Linux/Mac
# or
realtime_segmentation.exe  # Windows
```

### Runtime Controls

| Key | Action |
|-----|--------|
| **'1'** | Switch to Sequential backend |
| **'2'** | Switch to MPI backend |
| **'3'** | Switch to Multi-threaded backend |
| **'4'** | Switch to CUDA backend |
| **ESC** | Exit application |
| **K Slider** | Adjust cluster count (2-12) |

---

## 🔧 Configuration

### Algorithm Parameters

```cpp
// Default configuration values
const int k_min = 1;           // Minimum clusters
const int k_max = 12;          // Maximum clusters
const int sample_size = 2000;  // Coreset size
const float color_scale = 1.0f;   // Color feature scaling
const float spatial_scale = 0.5f; // Spatial feature scaling
```

### RCC Tree Settings

```cpp
const int max_levels = 8;      // Maximum tree height
const int default_sample = 2000; // Default coreset size
```

---

## 📈 Real-Time Performance Thresholds

| FPS Range | Classification | User Experience | Recommended Backends |
|-----------|----------------|-----------------|----------------------|
| **30+ FPS** | Excellent | Smooth real-time | CUDA, Multi-threaded (K≤8) |
| **20-30 FPS** | Good | Acceptable real-time | Multi-threaded, MPI (K≤6) |
| **10-20 FPS** | Fair | Noticeable lag | All backends (K≤4) |
| **<10 FPS** | Poor | Choppy playback | Sequential only (high K) |

---

## 🎯 Use Cases

### 🎬 Video Processing
- Real-time video segmentation for streaming applications
- Live broadcast effects and background replacement
- Content creation and video editing workflows

### 🤖 Computer Vision Research
- Baseline implementation for segmentation algorithms
- Performance benchmarking across different hardware
- Educational demonstrations of parallel computing concepts

### 🏥 Medical Imaging
- Real-time analysis of medical imagery
- Interactive segmentation for diagnostic applications
- High-throughput batch processing of medical data

### 🎮 Interactive Applications
- Real-time augmented reality applications
- Interactive art installations
- Gaming and entertainment systems

---

## 🛠️ Customization

### Adding New Backends

```cpp
// In clustering_backends.hpp
enum Backend { 
    BACKEND_SEQ = 0, 
    BACKEND_CUDA = 1, 
    BACKEND_THR = 2, 
    BACKEND_MPI = 3,
    BACKEND_CUSTOM = 4  // Your custom backend
};

// Implement your backend function
cv::Mat segmentFrameWithKMeans_custom(
    const cv::Mat& frame, int k, int sample_size,
    float color_scale, float spatial_scale);
```

### Tuning Performance

```cpp
// Adjust coreset sampling for speed vs quality trade-off
const int fast_sample_size = 1000;    // Faster, lower quality
const int quality_sample_size = 5000; // Slower, higher quality

// Modify feature scaling for different segmentation characteristics
const float color_emphasis = 2.0f;    // Emphasize color similarity
const float spatial_emphasis = 0.1f;  // De-emphasize spatial proximity
```

---

## 📂 Project Structure

```
realtime-parallel-kmeans-segmentation/
├── 📁 include/                      # Header files
│   ├── clustering.hpp               # Main clustering interface
│   ├── clustering_backends.hpp      # Backend implementations
│   ├── coreset.hpp                  # Coreset data structures
│   ├── rcc.hpp                      # RCC tree implementation
│   ├── utils.hpp                    # Utility functions
│   └── video_io.hpp                 # Video I/O interface
├── 📁 src/                          # Source files
│   ├── 📁 clustering/               # Backend implementations
│   │   ├── clustering_cuda.cu       # CUDA GPU backend
│   │   ├── clustering_entry.cpp     # Backend dispatcher
│   │   ├── clustering_mpi.cpp       # MPI distributed backend
│   │   ├── clustering_seq.cpp       # Sequential CPU backend
│   │   └── clustering_thr.cpp       # Multi-threaded backend
│   ├── coreset.cpp                  # Coreset algorithms
│   ├── main.cpp                     # Application entry point
│   ├── rcc.cpp                      # RCC tree implementation
│   ├── utils.cpp                    # Utility functions
│   └── video_io.cpp                 # Video I/O implementation
├── 📁 docs/                         # Documentation
│   ├── project__demo.gif            # Program demonstration GIF
├── 📁 docs/                         # Documentation
│   ├── algorithms.md                # Algorithm descriptions
│   ├── parallelization.md           # Synchronization details
│   └── performance.md               # Performance analysis
├── 📁 tests/                        # Test files
│   ├── test_clustering.cpp          # Clustering tests
│   ├── test_coreset.cpp             # Coreset tests
│   ├── test_rcc_.cpp                # RCC tree tests
│   ├── test_utils.cpp               # Utility tests
│   └── test_video_io_.cpp           # Video I/O tests
├── CMakeLists.txt                   # Build configuration
├── LICENSE                          # MIT License
└── README.md                        # This file
```

---

## 🔬 Technical Deep Dive

### Recursive Cached Coreset (RCC) Tree

The RCC tree enables efficient streaming K-means by:

1. **Leaf Insertion**: New frame coresets inserted with carry propagation
2. **Node Merging**: Weighted coreset combination with bounded size
3. **Root Computation**: Dynamic merging of all levels for comprehensive representation
4. **Memory Bounds**: Tree height limited to prevent unbounded growth

### Synchronization Strategies

- **Multi-threaded**: Lock-free design with const references and exclusive write regions
- **MPI**: Collective operations (MPI_Bcast, MPI_Gatherv) with hybrid OpenMP parallelization
- **CUDA**: Host-device synchronization with cudaDeviceSynchronize() barriers

---

## 🧪 Known Limitations

1. **Memory Requirements**: CUDA backend requires sufficient GPU memory for large images
2. **Network Dependency**: MPI performance varies with network latency and bandwidth
3. **K-value Scaling**: All backends show performance degradation with very high cluster counts
4. **Hardware Specific**: Optimal performance depends on specific hardware configuration

---

## 🔮 Possible Future Enhancements

- [ ] **Adaptive Coreset Sizing**: Dynamic adjustment based on image complexity
- [ ] **Additional Color Spaces**: Support for HSV, LAB, and other color representations
- [ ] **Temporal Coherence**: Frame-to-frame consistency improvements
- [ ] **Mobile Optimization**: ARM NEON and mobile GPU backend support
- [ ] **Cloud Integration**: Distributed processing across cloud instances

---

## 🙏 Acknowledgments

- **[OpenCV Team](https://opencv.org/)** – For comprehensive computer vision library and excellent documentation
- **[NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)** – For GPU computing platform and development tools
- **[Open MPI Project](https://www.open-mpi.org/)** – For high-performance message passing interface
- **[CMake Community](https://cmake.org/)** – For cross-platform build system
- **Research Community** – For foundational work on coreset algorithms and RCC trees

### Key References

- Feldman, D., Schmidt, M., & Sohler, C. (2013). *Turning big data into tiny data: Constant-size coresets for k-means, PCA and projective clustering*
- Bachem, O., Lucic, M., & Krause, A. (2017). *Practical coreset constructions for machine learning*

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 💡 Contact

Questions, feedback, or ideas? Reach out anytime at [sebastian.soptelea@proton.me](mailto:sebastian.soptelea@proton.me).
