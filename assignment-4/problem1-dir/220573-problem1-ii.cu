#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>

#define THRESHOLD (std::numeric_limits<double>::epsilon())

using std::cerr;
using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

#define cudaCheckError(ans)                                                    \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

const uint64_t N = (256);

// Version (ii): Shared memory tiling kernel
// Note: Shared memory size is determined by block dimensions at runtime
__global__ void shmem_kernel(double* in, double* out, int n) {
  // Shared memory with halos - uses maximum size, actual size depends on blockDim
  __shared__ double tile[10][10][10];
  
  int lx = threadIdx.x;
  int ly = threadIdx.y;
  int lz = threadIdx.z;
  
  // Global indices
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  int gz = blockIdx.z * blockDim.z + threadIdx.z;
  
  // Load main data into shared memory
  if (gx < n && gy < n && gz < n) {
    tile[lx + 1][ly + 1][lz + 1] = in[gx * n * n + gy * n + gz];
  }
  
  // Load halos - optimize by having boundary threads load extra data
  // Load x-direction halos
  if (lx == 0 && gx > 0) {
    tile[0][ly + 1][lz + 1] = in[(gx - 1) * n * n + gy * n + gz];
  }
  if (lx == blockDim.x - 1 && gx < n - 1) {
    tile[blockDim.x + 1][ly + 1][lz + 1] = in[(gx + 1) * n * n + gy * n + gz];
  }
  
  // Load y-direction halos
  if (ly == 0 && gy > 0) {
    tile[lx + 1][0][lz + 1] = in[gx * n * n + (gy - 1) * n + gz];
  }
  if (ly == blockDim.y - 1 && gy < n - 1) {
    tile[lx + 1][blockDim.y + 1][lz + 1] = in[gx * n * n + (gy + 1) * n + gz];
  }
  
  // Load z-direction halos
  if (lz == 0 && gz > 0) {
    tile[lx + 1][ly + 1][0] = in[gx * n * n + gy * n + (gz - 1)];
  }
  if (lz == blockDim.z - 1 && gz < n - 1) {
    tile[lx + 1][ly + 1][blockDim.z + 1] = in[gx * n * n + gy * n + (gz + 1)];
  }
  
  __syncthreads();

  // Compute stencil using shared memory 
  if (gx >= 1 && gx < n - 1 && gy >= 1 && gy < n - 1 && gz >= 1 && gz < n - 1) {
    out[gx * n * n + gy * n + gz] = 
      0.8 * (tile[lx][ly + 1][lz + 1] +      // i-1
             tile[lx + 2][ly + 1][lz + 1] +  // i+1
             tile[lx + 1][ly][lz + 1] +      // j-1
             tile[lx + 1][ly + 2][lz + 1] +  // j+1
             tile[lx + 1][ly + 1][lz] +      // k-1
             tile[lx + 1][ly + 1][lz + 2]);  // k+1
  }
}

__host__ void stencil(const double* in, double* out) {
  for (uint64_t i = 1; i < (N - 1); i++) {
    for (uint64_t j = 1; j < (N - 1); j++) {
      for (uint64_t k = 1; k < (N - 1); k++) {
        out[i * N * N + j * N + k] =
            0.8 * (in[(i - 1) * N * N + j * N + k] + 
                   in[(i + 1) * N * N + j * N + k] +
                   in[i * N * N + (j - 1) * N + k] + 
                   in[i * N * N + (j + 1) * N + k] +
                   in[i * N * N + j * N + (k - 1)] + 
                   in[i * N * N + j * N + (k + 1)]);
      }
    }
  }
}

__host__ void check_result(const double* w_ref, const double* w_opt,
                           const uint64_t size) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        double this_diff =
            std::fabs(w_ref[i * size * size + j * size + k] - 
                     w_opt[i * size * size + j * size + k]);
        if (this_diff > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

int main() {
  uint64_t NUM_ELEMS = (N * N * N);
  uint64_t SIZE_BYTES = (N * N * N) * sizeof(double);

  // Host memory allocation
  auto* h_in = new double[NUM_ELEMS];
  auto* h_out_cpu = new double[NUM_ELEMS];
  auto* h_out_gpu = new double[NUM_ELEMS];

  // Initialize input with random values
  srand(42);
  for (uint64_t i = 0; i < NUM_ELEMS; i++) {
    h_in[i] = static_cast<double>(rand()) / RAND_MAX;
  }
  std::fill_n(h_out_cpu, NUM_ELEMS, 0.0);
  std::fill_n(h_out_gpu, NUM_ELEMS, 0.0);

  // CPU baseline
  auto cpu_start = HR::now();
  stencil(h_in, h_out_cpu);
  auto cpu_end = HR::now();
  auto duration = duration_cast<microseconds>(cpu_end - cpu_start).count();
  cout << "Stencil time on CPU: " << duration << " us\n";

  // Device memory allocation
  double *d_in, *d_out;
  cudaCheckError(cudaMalloc(&d_in, SIZE_BYTES));
  cudaCheckError(cudaMalloc(&d_out, SIZE_BYTES));

  // Copy input to device
  cudaCheckError(cudaMemcpy(d_in, h_in, SIZE_BYTES, cudaMemcpyHostToDevice));

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float kernel_time = 0.0f;
  float h2d_time = 0.0f;
  float d2h_time = 0.0f;
  
  // Calculate CPU duration in ms for speedup comparison
  double cpu_duration_ms = duration / 1000.0;

  // Test different block sizes for shared memory kernel
  cout << "\n=== Block Size Performance Analysis ===\n";
  int block_sizes[] = {1, 2, 4, 8};
  for (int bs : block_sizes) {
    dim3 testBlock(bs, bs, bs);
    dim3 testGrid((N + bs - 1) / bs, (N + bs - 1) / bs, (N + bs - 1) / bs);
    
    cout << "\nBlock size " << bs << "x" << bs << "x" << bs << ":\n";
    
    // Time Host to Device transfer
    cudaCheckError(cudaMemset(d_out, 0, SIZE_BYTES));
    cudaEventRecord(start);
    cudaCheckError(cudaMemcpy(d_in, h_in, SIZE_BYTES, cudaMemcpyHostToDevice));
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&h2d_time, start, end);
    cout << "  H2D transfer time: " << h2d_time << " ms\n";
    
    // Time kernel execution only
    cudaEventRecord(start);
    shmem_kernel<<<testGrid, testBlock>>>(d_in, d_out, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&kernel_time, start, end);
    cout << "  Kernel execution time: " << kernel_time << " ms\n";
    
    // Time Device to Host transfer
    cudaEventRecord(start);
    cudaCheckError(cudaMemcpy(h_out_gpu, d_out, SIZE_BYTES, cudaMemcpyDeviceToHost));
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&d2h_time, start, end);
    cout << "  D2H transfer time: " << d2h_time << " ms\n";
    
    // Verify correctness
    cout << "  Correctness check: ";
    check_result(h_out_cpu, h_out_gpu, N);
    
    // Calculate speedups
    float total_time = h2d_time + kernel_time + d2h_time;
    float speedup = cpu_duration_ms / kernel_time;
    float overall_speedup = cpu_duration_ms / total_time;
    
    cout << "  Total time (transfers + kernel): " << total_time << " ms\n";
    cout << "  Kernel speedup vs CPU: " << speedup << "x\n";
    cout << "  Overall speedup vs CPU (including transfers): " << overall_speedup << "x\n";
  }

  // Cleanup
  cudaFree(d_in);
  cudaFree(d_out);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  
  delete[] h_in;
  delete[] h_out_cpu;
  delete[] h_out_gpu;

  return EXIT_SUCCESS;
}
