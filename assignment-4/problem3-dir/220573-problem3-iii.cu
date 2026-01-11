#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <iterator>
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>

using std::cerr;
using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;

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

#define BLOCK_SIZE_X 14
#define BLOCK_SIZE_Y 14
#define BLOCK_SIZE_Z 14

// Device kernel for grid search with UVM
__global__ void gridSearchUVMKernel(
    double *grid,      // Managed memory
    double *coeffs,    // Managed memory
    double kk,
    double *results,   // Managed memory
    int *count,        // Managed memory
    int *index,        // Managed memory
    int max_results)
{
  // Calculate 10D indices from 3D grid and block structure
  int idx1 = blockIdx.x;
  int idx2 = blockIdx.y;
  int idx3 = blockIdx.z;
  int idx4 = threadIdx.x;
  int idx5 = threadIdx.y;

  // Extract grid parameters - direct access to managed memory
  double dd1 = grid[0], dd2 = grid[1], dd3 = grid[2];
  double dd4 = grid[3], dd5 = grid[4], dd6 = grid[5];
  double dd7 = grid[6], dd8 = grid[7], dd9 = grid[8];
  double dd10 = grid[9], dd11 = grid[10], dd12 = grid[11];
  double dd13 = grid[12], dd14 = grid[13], dd15 = grid[14];
  double dd16 = grid[15], dd17 = grid[16], dd18 = grid[17];
  double dd19 = grid[18], dd20 = grid[19], dd21 = grid[20];
  double dd22 = grid[21], dd23 = grid[22], dd24 = grid[23];
  double dd25 = grid[24], dd26 = grid[25], dd27 = grid[26];
  double dd28 = grid[27], dd29 = grid[28], dd30 = grid[29];

  // Calculate loop bounds
  int s1 = floor((dd2 - dd1) / dd3);
  int s2 = floor((dd5 - dd4) / dd6);
  int s3 = floor((dd8 - dd7) / dd9);
  int s4 = floor((dd11 - dd10) / dd12);
  int s5 = floor((dd14 - dd13) / dd15);
  int s6 = floor((dd17 - dd16) / dd18);
  int s7 = floor((dd20 - dd19) / dd21);
  int s8 = floor((dd23 - dd22) / dd24);
  int s9 = floor((dd26 - dd25) / dd27);
  int s10 = floor((dd29 - dd28) / dd30);

  // Precompute coordinates for 6 dimensions mapped to grid/block
  double x1 = dd1 + idx1 * dd3;
  double x2 = dd4 + idx2 * dd6;
  double x3 = dd7 + idx3 * dd9;
  double x4 = dd10 + idx4 * dd12;
  double x5 = dd13 + idx5 * dd15;

  // Load coefficients and thresholds into registers - direct access to managed memory
  double c[10][10];
  double d[10];
  double ey[10];

  for (int i = 0; i < 10; i++)
  {
    for (int j = 0; j < 10; j++)
    {
      c[i][j] = coeffs[i * 12 + j];
    }
    d[i] = coeffs[i * 12 + 10];
    ey[i] = coeffs[i * 12 + 11];
  }

  // Calculate tolerance thresholds
  double e[10];
  for (int i = 0; i < 10; i++)
  {
    e[i] = kk * ey[i];
  }
  for(int r6 = 0; r6 < s6; r6++)
  {
    double x6 = dd16 + r6 * dd18;
    for (int r7 = 0; r7 < s7; r7++)
    {
      double x7 = dd19 + r7 * dd21;

      for (int r8 = 0; r8 < s8; r8++)
      {
        double x8 = dd22 + r8 * dd24;

        for (int r9 = 0; r9 < s9; r9++)
        {
          double x9 = dd25 + r9 * dd27;

          for (int r10 = 0; r10 < s10; r10++)
          {
            double x10 = dd28 + r10 * dd30;

            // Evaluate constraints
            bool valid = true;
            double x[10] = {x1, x2, x3, x4, x5, x6, x7, x8, x9, x10};

            for (int i = 0; i < 10 && valid; i++)
            {
              double sum = 0.0;
              for (int j = 0; j < 10; j++)
              {
                sum += c[i][j] * x[j];
              }
              double q = fabs(sum - d[i]);
              if (q > e[i])
              {
                valid = false;
              }
            }

            // Store result if valid - direct access to managed memory
            if (valid)
            {
              int idx = atomicAdd(count, 1);
              if (idx < max_results)
              {
                index[idx * 10 + 0] = idx1;
                index[idx * 10 + 1] = idx2;
                index[idx * 10 + 2] = idx3;
                index[idx * 10 + 3] = idx4;
                index[idx * 10 + 4] = idx5;
                index[idx * 10 + 5] = r6;
                index[idx * 10 + 6] = r7;
                index[idx * 10 + 7] = r8;
                index[idx * 10 + 8] = r9;
                index[idx * 10 + 9] = r10;
                for (int i = 0; i < 10; i++)
                {
                  results[idx * 10 + i] = x[i];
                }
              }
            }
          }
        }
      }
    }
  }
}

// Host entry
signed main()
{
  int i, j;
  i = 0;

  double h_grid[30];
  double h_coeffs[120];

  HRTimer start_dataread = HR::now();
  FILE *fp = fopen("problem3-dir/disp.txt", "r");
  if (fp == NULL)
  {
    cerr << "[ERROR] Could not open disp.txt file" << endl;
    return EXIT_FAILURE;
  }

  while (!feof(fp))
  {
    if (!fscanf(fp, "%lf", &h_coeffs[i]))
    {
      printf("[ERROR]: fscanf failed while reading problem3-dir/disp.txt\n");
      exit(EXIT_FAILURE);
    }
    i++;
  }
  fclose(fp);

  j = 0;
  fp = fopen("problem3-dir/grid.txt", "r");
  if (fp == NULL)
  {
    cerr << "[ERROR] Could not open grid.txt file" << endl;
    return EXIT_FAILURE;
  }
  while (!feof(fp))
  {
    if (!fscanf(fp, "%lf", &h_grid[j]))
    {
      printf("[ERROR]: fscanf failed while reading problem3-dir/grid.txt\n");
      exit(EXIT_FAILURE);
    }
    j++;
  }
  fclose(fp);

  HRTimer end_dataread = HR::now();

  double kk = 0.3;
  // Allocate managed memory (UVM)
  double *grid = nullptr, *coeffs = nullptr, *results = nullptr;
  int *index = nullptr;
  int *count = nullptr;

  const int MAX_RESULTS = 1000000;
  gpuAssert(cudaMallocManaged(&grid, 30 * sizeof(double)), __FILE__, __LINE__);
  gpuAssert(cudaMallocManaged(&coeffs, 120 * sizeof(double)), __FILE__, __LINE__);
  gpuAssert(cudaMallocManaged(&index, MAX_RESULTS * 10 * sizeof(int)), __FILE__, __LINE__);
  gpuAssert(cudaMallocManaged(&results, MAX_RESULTS * 10 * sizeof(double)), __FILE__, __LINE__);
  gpuAssert(cudaMallocManaged(&count, sizeof(int)), __FILE__, __LINE__);

  // Copy data to managed memory (CPU-side access)
  memcpy(grid, h_grid, 30 * sizeof(double));
  memcpy(coeffs, h_coeffs, 120 * sizeof(double));
  *count = 0;

  // Get device for memory hints
  int device;
  cudaGetDevice(&device);

  // Apply memory advise for read-only data
  gpuAssert(cudaMemAdvise(grid, 30 * sizeof(double), cudaMemAdviseSetReadMostly, device), __FILE__, __LINE__);
  gpuAssert(cudaMemAdvise(coeffs, 120 * sizeof(double), cudaMemAdviseSetReadMostly, device), __FILE__, __LINE__);

  // Set preferred location for frequently accessed data
  gpuAssert(cudaMemAdvise(grid, 30 * sizeof(double), cudaMemAdviseSetPreferredLocation, device), __FILE__, __LINE__);
  gpuAssert(cudaMemAdvise(coeffs, 120 * sizeof(double), cudaMemAdviseSetPreferredLocation, device), __FILE__, __LINE__);

  // Prefetch data to GPU
  HRTimer start_prefetch = HR::now();
  gpuAssert(cudaMemPrefetchAsync(grid, 30 * sizeof(double), device, 0), __FILE__, __LINE__);
  gpuAssert(cudaMemPrefetchAsync(coeffs, 120 * sizeof(double), device, 0), __FILE__, __LINE__);
  gpuAssert(cudaMemPrefetchAsync(results, MAX_RESULTS * 10 * sizeof(double), device, 0), __FILE__, __LINE__);
  gpuAssert(cudaMemPrefetchAsync(count, sizeof(int), device, 0), __FILE__, __LINE__);
  gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__);
  HRTimer end_prefetch = HR::now();

  int s1 = floor((h_grid[1] - h_grid[0]) / h_grid[2]);
  int s2 = floor((h_grid[4] - h_grid[3]) / h_grid[5]);
  int s3 = floor((h_grid[7] - h_grid[6]) / h_grid[8]);
  int s4 = floor((h_grid[10] - h_grid[9]) / h_grid[11]);
  int s5 = floor((h_grid[13] - h_grid[12]) / h_grid[14]);
  int s6 = floor((h_grid[16] - h_grid[15]) / h_grid[17]);
  int s7 = floor((h_grid[19] - h_grid[18]) / h_grid[20]);
  int s8 = floor((h_grid[22] - h_grid[21]) / h_grid[23]);
  int s9 = floor((h_grid[25] - h_grid[24]) / h_grid[26]);
  int s10 = floor((h_grid[28] - h_grid[27]) / h_grid[29]);

  dim3 gridDim(s1, s2, s3);
  dim3 blockDim(std::min(BLOCK_SIZE_X, std::max(1, s4)),
                std::min(BLOCK_SIZE_Y, std::max(1, s5)), 1);

  // Launch kernel
  // Time the kernel
  HRTimer start_compute = HR::now();
  gridSearchUVMKernel<<<gridDim, blockDim>>>(
      grid, coeffs, kk, results, count, index, MAX_RESULTS);
  gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
  gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__);
  HRTimer end_compute = HR::now();

  // Prefetch results back to CPU
  gpuAssert(cudaMemPrefetchAsync(count, sizeof(int), cudaCpuDeviceId, 0), __FILE__, __LINE__);
  gpuAssert(cudaMemPrefetchAsync(results, (*count) * 10 * sizeof(double), cudaCpuDeviceId, 0), __FILE__, __LINE__);
  gpuAssert(cudaMemPrefetchAsync(index, (*count) * 10 * sizeof(int), cudaCpuDeviceId, 0), __FILE__, __LINE__);
  gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__);

  // Access count directly (UVM)
  int h_count = *count;
  cout << "UVM version found " << h_count << " valid points\n";

  // Access results directly via UVM (no explicit copy needed)
  if (h_count > 0 && h_count <= MAX_RESULTS)
  {
    // Write to file - direct access to managed memory
    FILE *fp = fopen("problem3-dir/results-iii.txt", "w");
    
    // Create pairs of (index, result) for sorting
    std::vector<std::pair<std::array<int, 10>, std::array<double, 10>>> pairs;
    for (int i = 0; i < h_count; i++)
    {
      std::array<int, 10> idx_arr;
      std::array<double, 10> res_arr;
      for (int j = 0; j < 10; j++)
      {
        idx_arr[j] = index[i * 10 + j];
        res_arr[j] = results[i * 10 + j];
      }
      pairs.push_back({idx_arr, res_arr});
    }
    
    // Sort by indices (lexicographically)
    std::sort(pairs.begin(), pairs.end(), 
      [](const auto& a, const auto& b) {
        return a.first < b.first;
      });
    
    // Write sorted results to file
    for (const auto& p : pairs)
    {
      for (int j = 0; j < 10; j++)
      {
        fprintf(fp, "%lf\t", p.second[j]);
      }
      fprintf(fp, "\n");
    }
    
    fclose(fp);
  }

  auto dataread = duration_cast<microseconds>(end_dataread - start_dataread).count();
  auto prefetch = duration_cast<milliseconds>(end_prefetch - start_prefetch).count();
  auto compute = duration_cast<milliseconds>(end_compute - start_compute).count();
  auto total = duration_cast<milliseconds>(end_compute - start_dataread).count();
  cout << "[CPU] Data Read Time: " << dataread << " us" << endl;
  cout << "[GPU] UVM Prefetch Time: " << prefetch << " ms" << endl;
  cout << "[GPU] Kernel Compute Time: " << compute << " ms" << endl;
  cout << "[GPU] Total Time: " << total << " ms" << endl;

  // Cleanup - free managed memory
  gpuAssert(cudaFree(grid), __FILE__, __LINE__);
  gpuAssert(cudaFree(coeffs), __FILE__, __LINE__);
  gpuAssert(cudaFree(results), __FILE__, __LINE__);
  gpuAssert(cudaFree(index), __FILE__, __LINE__);
  gpuAssert(cudaFree(count), __FILE__, __LINE__);

  return EXIT_SUCCESS;
}
