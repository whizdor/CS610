#include <cassert>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <iterator>

using std::cerr;
using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const uint64_t N = (1ULL << 32);

#define cudaCheckError(ans)               \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

const int BLOCK_SIZE = 1024;
const int WARP_SIZE = 32;
__device__ __forceinline__ uint32_t warp_scan(uint32_t val)
{
  uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);

#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
  {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset)
      val += n;
  }

  return val;
}

__global__ void block_scan_kernel(const uint32_t *__restrict__ input,
                                  uint32_t *__restrict__ output,
                                  uint32_t *__restrict__ block_sums,
                                  uint64_t size)
{
  extern __shared__ uint32_t shared_data[];

  int tid = threadIdx.x;
  uint64_t global_id = blockIdx.x * blockDim.x + tid;

  uint32_t val = (global_id < size) ? input[global_id] : 0;

  int warp_id = tid / WARP_SIZE;
  int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  val = warp_scan(val);

  if ((tid & (WARP_SIZE - 1)) == (WARP_SIZE - 1))
  {
    shared_data[warp_id] = val;
  }
  __syncthreads();

  if (tid < num_warps)
  {
    uint32_t warp_sum = shared_data[tid];
    warp_sum = warp_scan(warp_sum);
    shared_data[tid] = warp_sum;
  }
  __syncthreads();

  if (warp_id > 0)
  {
    val += shared_data[warp_id - 1];
  }

  if (global_id < size)
  {
    output[global_id] = val;
  }

  if (tid == blockDim.x - 1 && block_sums != nullptr)
  {
    block_sums[blockIdx.x] = val;
  }
}

__global__ void add_block_sums_kernel(uint32_t *__restrict__ data,
                                      const uint32_t *__restrict__ block_sums,
                                      uint64_t size)
{
  uint64_t global_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (blockIdx.x > 0 && global_id < size)
  {
    data[global_id] += block_sums[blockIdx.x - 1];
  }
}

__host__ void cte_sum(const uint32_t *h_input, uint32_t *h_output, uint64_t size)
{
  int threads_per_block = BLOCK_SIZE;
  uint64_t num_blocks = (size + threads_per_block - 1) / threads_per_block;

  uint32_t *d_input, *d_output, *d_block_sums, *d_block_sums_scan;

  cudaCheckError(cudaMalloc(&d_input, size * sizeof(uint32_t)));
  cudaCheckError(cudaMalloc(&d_output, size * sizeof(uint32_t)));
  cudaCheckError(cudaMalloc(&d_block_sums, num_blocks * sizeof(uint32_t)));
  cudaCheckError(cudaMalloc(&d_block_sums_scan, num_blocks * sizeof(uint32_t)));

  cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
  cudaCheckError(cudaEventCreate(&start_total));
  cudaCheckError(cudaEventCreate(&stop_total));
  cudaCheckError(cudaEventCreate(&start_kernel));
  cudaCheckError(cudaEventCreate(&stop_kernel));

  cudaCheckError(cudaEventRecord(start_total));

  cudaCheckError(cudaMemcpy(d_input, h_input, size * sizeof(uint32_t),
                            cudaMemcpyHostToDevice));

  cudaCheckError(cudaEventRecord(start_kernel));
  int shared_mem_size = ((threads_per_block + WARP_SIZE - 1) / WARP_SIZE) * sizeof(uint32_t);
  block_scan_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
      d_input, d_output, d_block_sums, size);
  cudaCheckError(cudaGetLastError());

  if (num_blocks > 1)
  {
    uint64_t block_sum_blocks = (num_blocks + threads_per_block - 1) / threads_per_block;

    if (block_sum_blocks == 1)
    {
      block_scan_kernel<<<1, threads_per_block, shared_mem_size>>>(
          d_block_sums, d_block_sums_scan, nullptr, num_blocks);
    }
    else
    {
      uint32_t *d_temp;
      cudaCheckError(cudaMalloc(&d_temp, block_sum_blocks * sizeof(uint32_t)));

      block_scan_kernel<<<block_sum_blocks, threads_per_block, shared_mem_size>>>(
          d_block_sums, d_block_sums_scan, d_temp, num_blocks);

      block_scan_kernel<<<1, threads_per_block, shared_mem_size>>>(
          d_temp, d_temp, nullptr, block_sum_blocks);

      add_block_sums_kernel<<<block_sum_blocks, threads_per_block>>>(
          d_block_sums_scan, d_temp, num_blocks);

      cudaCheckError(cudaFree(d_temp));
    }

    add_block_sums_kernel<<<num_blocks, threads_per_block>>>(
        d_output, d_block_sums_scan, size);
  }

  cudaCheckError(cudaEventRecord(stop_kernel));
  cudaCheckError(cudaEventSynchronize(stop_kernel));

  cudaCheckError(cudaMemcpy(h_output, d_output, size * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));

  cudaCheckError(cudaEventRecord(stop_total));
  cudaCheckError(cudaEventSynchronize(stop_total));

  float kernel_time = 0, total_time = 0;
  cudaCheckError(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
  cudaCheckError(cudaEventElapsedTime(&total_time, start_total, stop_total));

  cout << "CTE - Kernel time (without memory copy): " << kernel_time << " ms\n";
  cout << "CTE - Total time (with memory copy): " << total_time << " ms\n";
  cout << "CTE - Bandwidth: " << (size * sizeof(uint32_t) * 3.0 / (total_time / 1000.0)) / 1e9
       << " GB/s\n";
  cudaCheckError(cudaFree(d_input));
  cudaCheckError(cudaFree(d_output));
  cudaCheckError(cudaFree(d_block_sums));
  cudaCheckError(cudaFree(d_block_sums_scan));
  cudaCheckError(cudaEventDestroy(start_total));
  cudaCheckError(cudaEventDestroy(stop_total));
  cudaCheckError(cudaEventDestroy(start_kernel));
  cudaCheckError(cudaEventDestroy(stop_kernel));
}

__host__ void uvm_sum(const uint32_t *h_input, uint32_t *h_output, uint64_t size)
{
  uint32_t *uvm_input, *uvm_output, *uvm_block_sums, *uvm_block_sums_scan;

  cudaCheckError(cudaMallocManaged(&uvm_input, size * sizeof(uint32_t)));
  cudaCheckError(cudaMallocManaged(&uvm_output, size * sizeof(uint32_t)));

  memcpy(uvm_input, h_input, size * sizeof(uint32_t));
  int threads_per_block = BLOCK_SIZE;
  uint64_t num_blocks = (size + threads_per_block - 1) / threads_per_block;

  cudaCheckError(cudaMallocManaged(&uvm_block_sums, num_blocks * sizeof(uint32_t)));
  cudaCheckError(cudaMallocManaged(&uvm_block_sums_scan, num_blocks * sizeof(uint32_t)));

  int device;
  cudaCheckError(cudaGetDevice(&device));
  cudaCheckError(cudaMemAdvise(uvm_input, size * sizeof(uint32_t),
                               cudaMemAdviseSetReadMostly, device));
  cudaCheckError(cudaMemAdvise(uvm_output, size * sizeof(uint32_t),
                               cudaMemAdviseSetPreferredLocation, device));
  cudaCheckError(cudaMemAdvise(uvm_block_sums, num_blocks * sizeof(uint32_t),
                               cudaMemAdviseSetPreferredLocation, device));
  cudaCheckError(cudaMemAdvise(uvm_block_sums_scan, num_blocks * sizeof(uint32_t),
                               cudaMemAdviseSetPreferredLocation, device));

  cudaCheckError(cudaMemPrefetchAsync(uvm_input, size * sizeof(uint32_t), device));
  cudaCheckError(cudaMemPrefetchAsync(uvm_output, size * sizeof(uint32_t), device));
  cudaCheckError(cudaMemPrefetchAsync(uvm_block_sums, num_blocks * sizeof(uint32_t), device));
  cudaCheckError(cudaMemPrefetchAsync(uvm_block_sums_scan, num_blocks * sizeof(uint32_t), device));

  cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
  cudaCheckError(cudaEventCreate(&start_total));
  cudaCheckError(cudaEventCreate(&stop_total));
  cudaCheckError(cudaEventCreate(&start_kernel));
  cudaCheckError(cudaEventCreate(&stop_kernel));

  cudaCheckError(cudaDeviceSynchronize());

  cudaCheckError(cudaEventRecord(start_total));
  cudaCheckError(cudaEventRecord(start_kernel));
  int shared_mem_size = ((threads_per_block + WARP_SIZE - 1) / WARP_SIZE) * sizeof(uint32_t);
  block_scan_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
      uvm_input, uvm_output, uvm_block_sums, size);
  cudaCheckError(cudaGetLastError());

  if (num_blocks > 1)
  {
    uint64_t block_sum_blocks = (num_blocks + threads_per_block - 1) / threads_per_block;

    if (block_sum_blocks == 1)
    {
      block_scan_kernel<<<1, threads_per_block, shared_mem_size>>>(
          uvm_block_sums, uvm_block_sums_scan, nullptr, num_blocks);
    }
    else
    {
      uint32_t *uvm_temp;
      cudaCheckError(cudaMallocManaged(&uvm_temp, block_sum_blocks * sizeof(uint32_t)));
      cudaCheckError(cudaMemAdvise(uvm_temp, block_sum_blocks * sizeof(uint32_t),
                                   cudaMemAdviseSetPreferredLocation, device));
      cudaCheckError(cudaMemPrefetchAsync(uvm_temp, block_sum_blocks * sizeof(uint32_t), device));

      block_scan_kernel<<<block_sum_blocks, threads_per_block, shared_mem_size>>>(
          uvm_block_sums, uvm_block_sums_scan, uvm_temp, num_blocks);

      block_scan_kernel<<<1, threads_per_block, shared_mem_size>>>(
          uvm_temp, uvm_temp, nullptr, block_sum_blocks);

      add_block_sums_kernel<<<block_sum_blocks, threads_per_block>>>(
          uvm_block_sums_scan, uvm_temp, num_blocks);

      cudaCheckError(cudaFree(uvm_temp));
    }

    add_block_sums_kernel<<<num_blocks, threads_per_block>>>(
        uvm_output, uvm_block_sums_scan, size);
  }

  cudaCheckError(cudaEventRecord(stop_kernel));
  cudaCheckError(cudaEventSynchronize(stop_kernel));

  cudaCheckError(cudaMemPrefetchAsync(uvm_output, size * sizeof(uint32_t), cudaCpuDeviceId));
  cudaCheckError(cudaDeviceSynchronize());

  cudaCheckError(cudaEventRecord(stop_total));
  cudaCheckError(cudaEventSynchronize(stop_total));

  float kernel_time = 0, total_time = 0;
  cudaCheckError(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
  cudaCheckError(cudaEventElapsedTime(&total_time, start_total, stop_total));

  cout << "UVM - Kernel time (without memory copy): " << kernel_time << " ms\n";
  cout << "UVM - Total time (with memory copy): " << total_time << " ms\n";
  cout << "UVM - Bandwidth: " << (size * sizeof(uint32_t) * 3.0 / (total_time / 1000.0)) / 1e9
       << " GB/s\n";
  memcpy(h_output, uvm_output, size * sizeof(uint32_t));

  cudaCheckError(cudaFree(uvm_input));
  cudaCheckError(cudaFree(uvm_output));
  cudaCheckError(cudaFree(uvm_block_sums));
  cudaCheckError(cudaFree(uvm_block_sums_scan));
  cudaCheckError(cudaEventDestroy(start_total));
  cudaCheckError(cudaEventDestroy(stop_total));
  cudaCheckError(cudaEventDestroy(start_kernel));
  cudaCheckError(cudaEventDestroy(stop_kernel));
}

__host__ void check_result(const uint32_t *w_ref, const uint32_t *w_opt,
                           const uint64_t size)
{
  bool correct = true;
  if (w_ref[0] != w_opt[0])
  {
    cout << "Mismatch at index 0: expected " << w_ref[0] << ", got " << w_opt[0] << "\n";
    correct = false;
  }

  if (w_ref[size - 1] != w_opt[size - 1])
  {
    cout << "Mismatch at index " << size - 1 << ": expected " << w_ref[size - 1]
         << ", got " << w_opt[size - 1] << "\n";
    correct = false;
  }

  for (uint64_t i = 0; i < std::min<uint64_t>(size, 1000); i += 100)
  {
    if (w_ref[i] != w_opt[i])
    {
      cout << "Mismatch at index " << i << ": expected " << w_ref[i]
           << ", got " << w_opt[i] << "\n";
      correct = false;
      break;
    }
  }

  if (correct)
  {
    cout << "Results match CPU reference (sampled check)\n";
  }
  else
  {
    cout << "ERROR: Results do NOT match CPU reference\n";
  }
}

__host__ void inclusive_prefix_sum(const uint32_t *input, uint32_t *output, uint64_t size)
{
  output[0] = input[0];
  for (uint64_t i = 1; i < size; i++)
  {
    output[i] = output[i - 1] + input[i];
  }
}

__host__ void cpu_sum(const uint32_t *h_input, uint32_t *h_output, uint64_t size)
{
  HRTimer start = HR::now();
  inclusive_prefix_sum(h_input, h_output, size);
  HRTimer end = HR::now();
  
  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << "CPU time: " << duration << " ms\n";
}

int main()
{
  cout << "Prefix Sum Implementation - Size: " << N << " elements ("
       << (N * sizeof(uint32_t)) / (1024.0 * 1024.0 * 1024.0) << " GB)\n\n";

  size_t free_mem, total_mem;
  cudaCheckError(cudaMemGetInfo(&free_mem, &total_mem));
  cout << "GPU Memory: " << free_mem / (1024.0 * 1024.0 * 1024.0) << " GB free / "
       << total_mem / (1024.0 * 1024.0 * 1024.0) << " GB total\n\n";

  auto *h_input = new uint32_t[N];
  for (uint64_t i = 0; i < N; i++)
  {
    h_input[i] = 1;
  }

  cout << "Computing CPU reference (sample)...\n";
  const uint64_t sample_size = std::min<uint64_t>(N, 10000);
  auto *h_output_cpu_sample = new uint32_t[sample_size];
  inclusive_prefix_sum(h_input, h_output_cpu_sample, sample_size);
  cout << "CPU sample result at index " << sample_size - 1 << ": "
       << h_output_cpu_sample[sample_size - 1] << "\n\n";

  auto *h_output_cpu = new uint32_t[N];
  for (uint64_t i = 0; i < N; i++)
  {
    h_output_cpu[i] = i + 1;
  }

  cout << "=== CPU Implementation ===\n";
  auto *h_output_cpu_full = new uint32_t[N];
  cpu_sum(h_input, h_output_cpu_full, N);
  check_result(h_output_cpu, h_output_cpu_full, N);
  cout << "\n";

  cout << "=== CTE Implementation ===\n";
  auto *h_output_cte = new uint32_t[N];
  cte_sum(h_input, h_output_cte, N);
  check_result(h_output_cpu, h_output_cte, N);
  cout << "\n";

  cout << "=== UVM Implementation ===\n";
  auto *h_output_uvm = new uint32_t[N];
  uvm_sum(h_input, h_output_uvm, N);
  check_result(h_output_cpu, h_output_uvm, N);
  cout << "\n";

  delete[] h_input;
  delete[] h_output_cpu;
  delete[] h_output_cpu_sample;
  delete[] h_output_cpu_full;
  delete[] h_output_cte;
  delete[] h_output_uvm;

  return EXIT_SUCCESS;
}
