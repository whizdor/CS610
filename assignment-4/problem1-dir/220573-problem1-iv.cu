#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>

#define THRESHOLD (std::numeric_limits<double>::epsilon())
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 8
#define TILE_SIZE_X 8
#define TILE_SIZE_Y 8
#define TILE_SIZE_Z 8

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

const uint64_t N = (512);

// Version (iv): Pinned memory version
__global__ void pinned_kernel(double* in, double* out, int n) {
  __shared__ double tile[TILE_SIZE_X + 2][TILE_SIZE_Y + 2][TILE_SIZE_Z + 2];
  
  int lx = threadIdx.x;
  int ly = threadIdx.y;
  int lz = threadIdx.z;
  
  int gz = blockIdx.z * blockDim.z + threadIdx.z;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  
  int idx_base = gx * n * n + gy * n;
  
  // Load main data
  if (gx < n && gy < n && gz < n) {
    tile[lx + 1][ly + 1][lz + 1] = in[idx_base + gz];
  }
  
  // Load all halos
  if (lx == 0 && gx > 0) {
    tile[0][ly + 1][lz + 1] = in[(gx - 1) * n * n + gy * n + gz];
  }
  if (lx == blockDim.x - 1 && gx < n - 1) {
    tile[TILE_SIZE_X + 1][ly + 1][lz + 1] = in[(gx + 1) * n * n + gy * n + gz];
  }
  if (ly == 0 && gy > 0) {
    tile[lx + 1][0][lz + 1] = in[gx * n * n + (gy - 1) * n + gz];
  }
  if (ly == blockDim.y - 1 && gy < n - 1) {
    tile[lx + 1][TILE_SIZE_Y + 1][lz + 1] = in[gx * n * n + (gy + 1) * n + gz];
  }
  if (lz == 0 && gz > 0) {
    tile[lx + 1][ly + 1][0] = in[idx_base + (gz - 1)];
  }
  if (lz == blockDim.z - 1 && gz < n - 1) {
    tile[lx + 1][ly + 1][TILE_SIZE_Z + 1] = in[idx_base + (gz + 1)];
  }
  
  __syncthreads();
  
  // Compute stencil with optimized memory access
  if (gx >= 1 && gx < n - 1 && gy >= 1 && gy < n - 1 && gz >= 1 && gz < n - 1) {
    double result = tile[lx][ly + 1][lz + 1] + tile[lx + 2][ly + 1][lz + 1] +
                   tile[lx + 1][ly][lz + 1] + tile[lx + 1][ly + 2][lz + 1] +
                   tile[lx + 1][ly + 1][lz] + tile[lx + 1][ly + 1][lz + 2];
    out[idx_base + gz] = 0.8 * result;
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

  // Configure kernel launch parameters
  dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                (N + blockSize.y - 1) / blockSize.y,
                (N + blockSize.z - 1) / blockSize.z);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float kernel_time = 0.0f;
  
  // Calculate CPU duration in ms for speedup comparison
  double cpu_duration_ms = duration / 1000.0;

  // Version (iv): Pinned memory version
  // Allocate pinned memory
  double *h_in_pinned, *h_out_pinned;
  cudaCheckError(cudaHostAlloc(&h_in_pinned, SIZE_BYTES, cudaHostAllocDefault));
  cudaCheckError(cudaHostAlloc(&h_out_pinned, SIZE_BYTES, cudaHostAllocDefault));
  
  // Copy data to pinned memory
  memcpy(h_in_pinned, h_in, SIZE_BYTES);
  
  cout << "\n=== Version (iv): Pinned Memory Performance ===\n";
  
  // Time pinned memory transfers
  float transfer_time_pinned = 0.0f;
  cudaCheckError(cudaMemset(d_out, 0, SIZE_BYTES));
  cudaEventRecord(start);
  cudaCheckError(cudaMemcpy(d_in, h_in_pinned, SIZE_BYTES, cudaMemcpyHostToDevice));
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&transfer_time_pinned, start, end);
  cout << "Pinned H2D transfer time: " << transfer_time_pinned << " ms\n";
  
  // Time regular memory transfers for comparison
  float transfer_time_regular = 0.0f;
  cudaEventRecord(start);
  cudaCheckError(cudaMemcpy(d_in, h_in, SIZE_BYTES, cudaMemcpyHostToDevice));
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&transfer_time_regular, start, end);
  cout << "Regular H2D transfer time: " << transfer_time_regular << " ms\n";
  cout << "Transfer speedup (pinned vs regular): " << transfer_time_regular / transfer_time_pinned << "x\n";
  
  // Time kernel execution only
  cudaCheckError(cudaMemset(d_out, 0, SIZE_BYTES));
  cudaEventRecord(start);
  pinned_kernel<<<gridSize, blockSize>>>(d_in, d_out, N);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  cout << "\nKernel execution time: " << kernel_time << " ms";
  
  // Copy result back using pinned memory
  float d2h_time_pinned = 0.0f;
  cudaEventRecord(start);
  cudaCheckError(cudaMemcpy(h_out_pinned, d_out, SIZE_BYTES, cudaMemcpyDeviceToHost));
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&d2h_time_pinned, start, end);
  cout << "\nPinned D2H transfer time: " << d2h_time_pinned << " ms\n";
  
  // Verify correctness
  cout << "Correctness check: ";
  check_result(h_out_cpu, h_out_pinned, N);
  
  // Report total time and speedups
  float total_time = transfer_time_pinned + kernel_time + d2h_time_pinned;
  cout << "\nTotal time (pinned transfers + kernel): " << total_time << " ms\n";
  float speedup = cpu_duration_ms / kernel_time;
  cout << "Kernel speedup vs CPU: " << speedup << "x\n";
  float overall_speedup = cpu_duration_ms / total_time;
  cout << "Overall speedup vs CPU (including transfers): " << overall_speedup << "x\n";

  // Cleanup
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFreeHost(h_in_pinned);
  cudaFreeHost(h_out_pinned);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  
  delete[] h_in;
  delete[] h_out_cpu;
  delete[] h_out_gpu;

  return EXIT_SUCCESS;
}
