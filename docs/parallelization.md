# Parallelization

## Overview

This project supports multiple parallelization backends for K-Means image segmentation, enabling efficient use of modern hardware: multi-core CPUs, distributed systems, and GPUs. Each backend uses a different parallelization strategy and synchronization mechanism, described below.

---

## Multi-threaded (C++ std::thread / OpenMP)

### Approach

- The image is divided into horizontal chunks (blocks of rows), one per thread.
- Each thread processes its assigned rows independently, reading from the input image and writing to a unique region of the output image.
- The number of threads is determined by hardware concurrency or set to a default value.

### Synchronization

- No explicit synchronization is needed for the main computation, as each thread writes to a separate region of the output.
- Threads are joined at the end to ensure all work is complete before returning the result.

### Code Example
// Launch worker threads for each chunk of rows for (unsigned int t = 0; t < numThreads; ++t) { workers.emplace_back(processRows, ...); } // Wait for all threads to finish for (auto& th : workers) th.join();


---

## Distributed (MPI)

### Approach

- The image is divided into blocks of rows, distributed across MPI processes (ranks).
- Rank 0 computes the K-Means cluster centers and broadcasts them to all ranks.
- Each rank processes its assigned rows, assigning pixels to clusters independently.
- Results are gathered at rank 0 using `MPI_Gatherv` to assemble the final segmented image.

### Synchronization

- **MPI_Bcast:** Used to broadcast cluster centers from rank 0 to all other ranks, ensuring all processes use the same centers.
- **MPI_Gatherv:** Used to gather the processed image chunks from all ranks to rank 0.
- No explicit point-to-point synchronization is needed; collective operations ensure data consistency.

### Hybrid Parallelism

- Within each MPI rank, OpenMP or similar threading can be used to further parallelize the assignment step, leveraging all CPU cores on each node.

### Code Example
// Broadcast cluster centers MPI_Bcast(flatCenters.data(), k * 5, MPI_FLOAT, 0, MPI_COMM_WORLD); // Parallel for over assigned rows (OpenMP) #pragma omp parallel for schedule(dynamic) for (int r = startRow; r < endRow; ++r) { ... } // Gather results at rank 0 MPI_Gatherv(localOut.data, ..., out.data, ..., 0, MPI_COMM_WORLD);


---

## GPU-accelerated (CUDA)

### Approach

- Cluster centers are computed on the CPU and transferred to the GPU.
- The input image is copied to the GPU.
- A CUDA kernel is launched, with one thread per pixel, to assign pixels to clusters in parallel.
- The segmented image is copied back to the CPU.

### Synchronization

- **Kernel Launch:** All threads execute in parallel; each thread processes one pixel.
- **cudaDeviceSynchronize():** Ensures all threads have completed before copying results back to the host.
- No explicit synchronization is needed within the kernel, as each thread works independently.

### Code Example
assignPixelsKernel<<<blocks, threadsPerBlock>>>(...); cudaDeviceSynchronize(); cudaMemcpy(out.data, d_output, ...);


---

## Summary Table

| Backend      | Parallelization Strategy         | Synchronization                |
|--------------|----------------------------------|-------------------------------|
| Threads      | Row blocks per thread            | Thread join                   |
| OpenMP       | Row blocks per thread (pragma)   | Implicit (OpenMP runtime)     |
| MPI          | Row blocks per process           | MPI_Bcast, MPI_Gatherv        |
| CUDA         | Pixel per thread (GPU)           | cudaDeviceSynchronize         |

---

## Design Considerations

- **Data Partitioning:** All parallel backends partition the image by rows for cache efficiency and simplicity.
- **Read-Only Sharing:** Input image and cluster centers are read-only for all threads/processes, avoiding race conditions.
- **Output Partitioning:** Each thread/process writes to a unique region of the output image, eliminating the need for locks.
- **Collective Operations:** MPI uses collective communication for synchronization and data movement.

---

## References

- [OpenMP Documentation](https://www.openmp.org/)
- [MPI Standard](https://www.mpi-forum.org/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
