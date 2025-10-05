# Parallelization and Synchronization

## Overview

This project supports multiple parallelization backends for K-Means image segmentation, enabling efficient use of modern hardware: multi-core CPUs, distributed systems, and GPUs. Each backend uses different parallelization strategies and synchronization mechanisms to ensure thread safety, data consistency, and optimal performance.

---

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
    std::cref(frame),      // Read-only: multiple threads can safely read
    std::ref(out),         // Write-only: each thread writes to different rows
    std::cref(centers),    // Read-only: shared cluster centers
    rStart, rEnd,          // Unique row range per thread
    color_scale, spatial_scale);
```

**Key Safety Features:**
- **Const references** for shared read-only data prevent accidental modifications
- **Non-overlapping write regions** eliminate the need for output synchronization
- **Value-based parameters** (row ranges, scales) are copied per thread

---

## Distributed (MPI + OpenMP Hybrid)

### Approach

- **Process-level parallelism**: Image divided into row blocks across MPI processes (ranks)
- **Thread-level parallelism**: Each MPI rank uses OpenMP for additional parallelization within its assigned rows
- **Master-worker pattern**: Rank 0 computes cluster centers and coordinates data distribution

### Synchronization Strategy

**Inter-Process Synchronization (MPI Level):**

1. **Cluster Center Distribution:**
   ```cpp
   // Rank 0 computes centers, others prepare to receive
   if (rank != 0) flatCenters.resize(k * 5);
   MPI_Bcast(flatCenters.data(), k * 5, MPI_FLOAT, 0, MPI_COMM_WORLD);
   ```
   - **MPI_Bcast**: Collective operation ensuring all ranks receive identical cluster centers
   - **Synchronization barrier**: All ranks must reach this point before proceeding
   - **Data consistency**: Guarantees all processes use the same clustering parameters

2. **Result Aggregation:**
   ```cpp
   // Calculate data layout for gathering
   std::vector<int> recvCounts(size), displs(size);
   // Gather processed chunks from all ranks
   MPI_Gatherv(localOut.data, ..., out.data, recvCounts.data(), 
               displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
   ```
   - **MPI_Gatherv**: Variable-length gather handling uneven row distribution
   - **Displacement arrays**: Ensure correct placement of each rank's contribution
   - **Memory layout consistency**: Maintains proper BGR channel ordering

**Intra-Process Synchronization (OpenMP Level):**
```cpp
#pragma omp parallel for schedule(dynamic)
for (int r = startRow; r < endRow; ++r) {
    // Thread-safe pixel processing within each rank
}
```
- **Dynamic scheduling**: Automatic load balancing across threads within each rank
- **Implicit barrier**: OpenMP automatically synchronizes threads at the end of the parallel region
- **Thread-local variables**: Each thread processes distinct rows, avoiding race conditions

**Hybrid Synchronization Benefits:**
- **Two-tier parallelism**: Distributes work across machines (MPI) and cores (OpenMP)
- **Minimal communication**: Only cluster centers and final results are communicated
- **Fault tolerance**: Each rank operates independently during computation phase

---

## GPU-accelerated (CUDA)

### Approach

- **Host-device workflow**: Cluster centers computed on CPU, pixel assignment on GPU
- **Massive parallelism**: One CUDA thread per pixel for maximum throughput
- **Memory hierarchy optimization**: Efficient data transfer and kernel execution

### Synchronization Strategy

**Host-Device Synchronization:**

1. **Memory Transfer Synchronization:**
   ```cpp
   // Host-to-device transfers (blocking operations)
   cudaMemcpy(d_input, frame.data, ..., cudaMemcpyHostToDevice);
   cudaMemcpy(d_centers, flatCenters.data(), ..., cudaMemcpyHostToDevice);
   ```
   - **Blocking transfers**: Ensures data is fully copied before kernel launch
   - **Memory coherence**: Host modifications visible to device after transfer

2. **Kernel Execution Synchronization:**
   ```cpp
   // Launch kernel asynchronously
   assignPixelsKernel<<<blocks, threadsPerBlock>>>(
       d_input, d_output, cols, rows, d_centers, k, color_scale, spatial_scale);
   
   // Error checking and synchronization
   cudaGetLastError();        // Check for launch errors
   cudaDeviceSynchronize();   // Wait for all threads to complete
   ```
   - **Asynchronous launch**: Kernel execution overlaps with host operations where possible
   - **cudaDeviceSynchronize()**: Explicit barrier ensuring all GPU threads complete
   - **Error propagation**: Checks for kernel launch failures

3. **Result Retrieval:**
   ```cpp
   // Device-to-host transfer (blocking)
   cudaMemcpy(out.data, d_output, ..., cudaMemcpyDeviceToHost);
   ```
   - **Blocking retrieval**: Ensures all GPU work completes before host access

**GPU Thread Synchronization:**

Within the CUDA kernel:
```cuda
__global__ void assignPixelsKernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Unique thread ID
    if (idx >= total) return;  // Bounds checking
    
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

---

## Performance and Scalability Characteristics

### Synchronization Overhead Analysis

| Backend | Sync Overhead | Scalability | Best Use Case |
|---------|---------------|-------------|---------------|
| **std::thread** | Minimal (join only) | CPU cores | Local multi-core systems |
| **MPI+OpenMP** | Moderate (broadcast/gather) | Distributed nodes | HPC clusters, large images |
| **CUDA** | Low (device sync) | GPU cores | High-throughput applications |

### Design Principles

**Common Synchronization Patterns:**
1. **Data partitioning**: All backends partition work to minimize synchronization needs
2. **Read-only sharing**: Cluster centers and input data shared without locks
3. **Exclusive write regions**: Each processing unit writes to unique output areas
4. **Barrier synchronization**: Explicit synchronization only at completion boundaries

**Thread Safety Guarantees:**
- **No race conditions**: Careful data partitioning eliminates concurrent writes
- **Memory consistency**: Appropriate synchronization ensures visibility of shared data
- **Resource safety**: Proper cleanup and error handling in all backends

---

## References

- [C++ Threading Library](https://en.cppreference.com/w/cpp/thread)
- [OpenMP Specification](https://www.openmp.org/specifications/)
- [MPI Standard Documentation](https://www.mpi-forum.org/docs/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Memory Models and Synchronization](https://en.cppreference.com/w/cpp/atomic/memory_order)
