# Parallelization and Synchronization

## Overview

This project supports multiple parallelization backends for K-Means image segmentation, enabling efficient use of modern hardware: multi-core CPUs, distributed systems, and GPUs. Each backend uses different parallelization strategies and synchronization mechanisms to ensure thread safety, data consistency, and optimal performance.

-----

## Multi-threaded (C++ std::thread)

### Approach

  - The image is divided into horizontal chunks (blocks of rows), one per thread.
  - Each thread processes its assigned rows independently, reading from the input image and writing to a unique region of the output image.
  - The number of threads is automatically determined using `std::thread::hardware_concurrency()` (typically the number of CPU cores), with a fallback to 4 threads.

### Synchronization Strategy

**Lock-Free Design:**

  - **No explicit locks or mutexes** are needed because each thread operates on disjoint data regions.
  - **Read-only sharing**: Input frame and cluster centers are shared read-only across all threads using `std::cref()`.
  - **Exclusive write access**: Each thread writes to a unique range of output rows, eliminating race conditions.

**Thread Coordination:**

  - **Thread creation**: Threads are launched using `std::thread::emplace_back()` with function parameters passed by reference.
  - **Barrier synchronization**: `th.join()` is called on each thread to ensure all processing completes before returning results.
  - **Load balancing**: The last thread may process extra rows if the total rows don't divide evenly by thread count.

### Memory Safety Guarantees

```cpp
// Safe concurrent access patterns:
workers.emplace_back(processRows,
    std::cref(frame),       // Read-only: multiple threads can safely read
    std::ref(out),          // Write-only: each thread writes to different rows
    std::cref(centers),     // Read-only: shared cluster centers
    rStart, rEnd,           // Unique row range per thread
    color_scale, spatial_scale);
```

**Key Safety Features:**

  - **Const references** for shared read-only data prevent accidental modifications
  - **Non-overlapping write regions** eliminate the need for output synchronization
  - **Value-based parameters** (row ranges, scales) are copied per thread

-----

## Distributed (MPI Hybrid)

This backend implements a **Master-Worker pattern** using **MPI for distributed processing** and **OpenMP for intra-process multi-threading** (hybrid parallelism).

### Approach

  - **Process-level parallelism (MPI)**: The image is divided into contiguous blocks of rows, and each block is assigned to a unique MPI process (rank).
  - **Thread-level parallelism (OpenMP)**: Within its assigned row block, each MPI rank uses OpenMP's `#pragma omp parallel for` to further parallelize the pixel processing across its available CPU cores.
  - **Master-Worker**: **Rank 0** is the designated *master*, responsible for computing the cluster centers, distributing them, and aggregating the final result.

### Implementation Details

| Step | Location | Parallelism/Operation | Description |
| :--- | :--- | :--- | :--- |
| **1. Center Computation** | **Rank 0** only | Single-threaded | Master computes the initial **K-Means cluster centers** using the `computeKMeansCenters` utility. |
| **2. Center Distribution** | All Ranks | **MPI\_Bcast** | All processes wait at an `MPI_Barrier`, then Rank 0 broadcasts the flattened $k \times 5$ cluster centers to all other ranks, ensuring global consistency. |
| **3. Local Data Preparation** | All Ranks | **OpenMP** | Each rank determines its unique range of rows, extracts that chunk from the input image, and flattens it into a local vector (`localFlat`) using an **OpenMP dynamic schedule** for efficient internal thread distribution. |
| **4. Local Processing** | All Ranks | **OpenMP** | Each rank executes the pixel assignment for its local data chunk. The loop iterates over local rows and uses the broadcasted centers to segment the local pixels, storing the result in `localResult`. This step also uses an **OpenMP dynamic schedule**. |
| **5. Result Aggregation** | All Ranks $\rightarrow$ **Rank 0** | **MPI\_Gatherv** | Ranks calculate the count and displacement necessary for gathering. All ranks send their `localResult` to the master (Rank 0), which uses `MPI_Gatherv` to reconstruct the final, complete image (`out.data`) from the variable-sized chunks. |

### Synchronization Strategy

The hybrid strategy employs synchronization at two distinct levels:

#### Inter-Process Synchronization (MPI Level)

1.  **Barrier Synchronization:**

    ```cpp
    MPI_Barrier(MPI_COMM_WORLD);
    ```

    An explicit **MPI\_Barrier** is used *before* the broadcast to ensure all processes have reached the same point before the critical data distribution begins, maintaining order.

2.  **Cluster Center Distribution (Collective Communication):**

    ```cpp
    MPI_Bcast(flatCenters.data(), k * 5, MPI_FLOAT, 0, MPI_COMM_WORLD);
    ```

      - **MPI\_Bcast**: A collective operation that ensures every rank receives the identical set of cluster centers computed by Rank 0. This is the **primary synchronization point** for the clustering parameters.

3.  **Result Aggregation (Collective Communication):**

    ```cpp
    MPI_Gatherv(localResult.data(), localResult.size(), MPI_UNSIGNED_CHAR,
        (rank == 0 ? out.data : nullptr), recvCounts.data(), displs.data(),
        MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    ```

      - **MPI\_Gatherv**: A collective synchronization point where all worker ranks simultaneously transmit their computed image segments back to the master (Rank 0), which collects and correctly places them into the final output image using calculated `recvCounts` and `displs` arrays.

#### Intra-Process Synchronization (OpenMP Level)

```cpp
#pragma omp parallel for schedule(dynamic)
// ... loops for flattening input and processing local chunk
```

  - **Thread Safety:** The OpenMP parallel regions are inherently **thread-safe** because each thread processes a unique, disjoint set of pixels (rows) within the rank's assigned chunk.
  - **Implicit Barrier:** OpenMP inserts an **implicit barrier** at the end of the `#pragma omp parallel for` regions, ensuring all threads within a rank complete their work before the rank proceeds to the next MPI operation (e.g., `MPI_Gatherv`).
  - **Dynamic Scheduling:** The `schedule(dynamic)` clause provides **automatic load balancing** across the threads within each MPI rank.

-----

## Real-time Webcam Feed: Synchronization and Communication

The `showWebcamFeed` function manages the master-worker communication necessary for processing a continuous video stream using MPI. **Rank 0 (Master)** handles the webcam capture and user interface, while **all ranks** participate in processing the frame. The primary mechanism for data sharing and synchronization is $\text{MPI\_Bcast}$.

### MPI Broadcasts in Webcam Loop

In each iteration of the main processing loop, Rank 0 captures the new frame and then broadcasts the necessary data and control information to all other worker ranks to ensure they are processing the same frame with the same parameters.

| Variable/Data | Purpose | MPI Call | Description |
| :--- | :--- | :--- | :--- |
| **`rows`, `cols`, `type`** | Frame Dimensions | `MPI_Bcast(&var, 1, MPI_INT, 0, ...)` | **Rank 0** sends the dimensions and type of the captured frame (e.g., $480 \times 640$, $\text{CV\_8UC3}$) to all ranks. This allows worker ranks to correctly allocate memory for the incoming frame data. |
| **`frame.data`** | Frame Image Data | `MPI_Bcast(frame.data, rows * frame.step, MPI_BYTE, 0, ...)` | **Rank 0** sends the entire raw byte data of the captured frame to all ranks. This is a large, collective transfer, ensuring all workers have an identical copy of the frame to segment. |
| **`k`** | Cluster Count | `MPI_Bcast(&k, 1, MPI_INT, 0, ...)` | **Rank 0** sends the current number of clusters, $k$, read from the trackbar. This ensures all ranks use the same $\text{k}$ value during the $\text{K-Means}$ segmentation process. |
| **`backend`** | Processing Backend | `MPI_Bcast(&backend, 1, MPI_INT, 0, ...)` | **Rank 0** sends the currently selected processing backend (e.g., $\text{MPI}, \text{CUDA}$) to ensure all ranks execute the same path or know which backend is active (especially if the `segmentFrameWithKMeans` function uses this to route the call). |
| **`stopFlag`** | Loop Termination | `MPI_Bcast(&stopFlag, 1, MPI_INT, 0, ...)` | **Rank 0** broadcasts a flag to signal that the user has closed the window or pressed ESC. This allows the worker ranks to gracefully exit the infinite loop and terminate the program alongside the master. |

### Synchronization Rationale

  - **Consistency**: The use of $\text{MPI\_Bcast}$ for key parameters ($k$, frame dimensions) and the frame data itself guarantees **data consistency** across all distributed processes in every single frame, which is crucial for correct segmentation.
  - **Implicit Synchronization**: $\text{MPI\_Bcast}$ is a **collective communication** operation, acting as an **implicit barrier**. All processes must call $\text{MPI\_Bcast}$ and wait until the data transfer from the root (Rank 0) is complete before proceeding to the segmentation step. This ensures that no worker begins processing a frame before it has fully received the image data and the current $k$ value.

-----

## GPU-accelerated (CUDA)

### Approach

  - **Host-device workflow**: Cluster centers computed on CPU, pixel assignment on GPU
  - **Massive parallelism**: One CUDA thread per pixel for maximum throughput
  - **Memory hierarchy optimization**: Efficient data transfer and kernel execution

### Synchronization Strategy

**Host-Device Synchronization:**

1.  **Memory Transfer Synchronization:**

    ```cpp
    // Host-to-device transfers (blocking operations)
    cudaMemcpy(d_input, frame.data, ..., cudaMemcpyHostToDevice);
    cudaMemcpy(d_centers, flatCenters.data(), ..., cudaMemcpyHostToDevice);
    ```

      - **Blocking transfers**: Ensures data is fully copied before kernel launch
      - **Memory coherence**: Host modifications visible to device after transfer

2.  **Kernel Execution Synchronization:**

    ```cpp
    // Launch kernel asynchronously
    assignPixelsKernel<<<blocks, threadsPerBlock>>>(...);

    // Error checking and synchronization
    cudaGetLastError();
    cudaDeviceSynchronize();  // Wait for all threads to complete
    ```

      - **Asynchronous launch**: Kernel execution overlaps with host operations where possible
      - **cudaDeviceSynchronize()**: Explicit barrier ensuring all GPU threads complete
      - **Error propagation**: Checks for kernel launch failures

3.  **Result Retrieval:**

    ```cpp
    // Device-to-host transfer (blocking)
    cudaMemcpy(out.data, d_output, ..., cudaMemcpyDeviceToHost);
    ```

      - **Blocking retrieval**: Ensures all GPU work completes before host access

**GPU Thread Synchronization:**

Within the CUDA kernel:

```cuda
__global__ void assignPixelsKernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Each thread processes one pixel independently
    // No inter-thread communication or synchronization needed
}
```

**Key CUDA Synchronization Features:**

  - **Lock-free design**: Each thread processes a unique pixel, no shared writes
  - **Automatic memory consistency**: GPU hardware ensures coherent memory access
  - **Block-level parallelism**: Threads organized in blocks for efficient execution
  - **Implicit synchronization**: Block and grid completion handled by CUDA runtime

**Memory Management Synchronization:**

  - **Resource cleanup**: `cudaFree()` operations ensure proper device memory deallocation
  - **Error handling**: Comprehensive error checking at each synchronization point

-----

## Performance and Scalability Characteristics

### Synchronization Overhead Analysis

| Backend | Sync Overhead | Scalability | Best Use Case |
| :--- | :--- | :--- | :--- |
| **std::thread** | Minimal (join only) | CPU cores | Local multi-core systems |
| **MPI Hybrid** | Moderate (broadcast/gather) | Distributed nodes | HPC clusters, large images |
| **CUDA** | Low (device sync) | GPU cores | High-throughput applications |

### Design Principles

**Common Synchronization Patterns:**

1.  **Data partitioning**: All backends partition work to minimize synchronization needs
2.  **Read-only sharing**: Cluster centers and input data shared without locks
3.  **Exclusive write regions**: Each processing unit writes to unique output areas
4.  **Barrier synchronization**: Explicit synchronization only at completion boundaries

**Thread Safety Guarantees:**

  - **No race conditions**: Careful data partitioning eliminates concurrent writes
  - **Memory consistency**: Appropriate synchronization ensures visibility of shared data
  - **Resource safety**: Proper cleanup and error handling in all backends

-----

## References

  - [C++ Threading Library](https://en.cppreference.com/w/cpp/thread)
  - [OpenMP Specification](https://www.openmp.org/specifications/)
  - [MPI Standard Documentation](https://www.mpi-forum.org/docs/)
  - [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
  - [Memory Models and Synchronization](https://en.cppreference.com/w/cpp/atomic/memory_order)