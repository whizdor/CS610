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

// Constant memory for grid parameters
__constant__ double const_grid_params[30];
__constant__ double const_constraint_matrix[120];
__constant__ double const_tolerance_factor;
__constant__ double const_error_thresholds[10];
__constant__ long long const_dimension_sizes[10];

// Global device counter for results
__device__ unsigned long long global_result_counter = 0;

// Maximum capacity for storing results
constexpr unsigned long long RESULT_CAPACITY = 40000ULL;

// Kernel for 1D flattened grid search
__global__ void flattenedGridSearchKernel(
    double *output_results,
    long long starting_offset,
    long long num_iterations)
{
  // Compute global thread index
  long long thread_global_id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  
  // Boundary check
  if (thread_global_id >= num_iterations) {
    return;
  }
  
  // Compute actual iteration index
  long long iteration_id = starting_offset + thread_global_id;
  
  // Extract dimension sizes from constant memory
  long long dim_size[10];
  #pragma unroll
  for (int k = 0; k < 10; k++) {
    dim_size[k] = const_dimension_sizes[k];
  }
  
  // Decompose flattened index into 10D coordinates
  long long coord[10];
  long long temp_idx = iteration_id;
  long long divisor = 1;
  
  // Calculate coordinates in reverse order
  for (int k = 9; k >= 0; k--) {
    if (k == 9) {
      coord[k] = temp_idx % dim_size[k];
      temp_idx /= dim_size[k];
    } else {
      coord[k] = temp_idx % dim_size[k];
      temp_idx /= dim_size[k];
    }
  }
  
  // Compute actual coordinate values
  double position[10];
  #pragma unroll
  for (int k = 0; k < 10; k++) {
    position[k] = const_grid_params[3 * k] + coord[k] * const_grid_params[3 * k + 2];
  }
  
  // Evaluate all constraint equations
  double constraint_residuals[10];
  bool all_constraints_satisfied = true;
  
  #pragma unroll
  for (int eq = 0; eq < 10; eq++) {
    double linear_combination = 0.0;
    
    #pragma unroll
    for (int var = 0; var < 10; var++) {
      linear_combination += const_constraint_matrix[eq * 12 + var] * position[var];
    }
    
    constraint_residuals[eq] = fabs(linear_combination - const_constraint_matrix[eq * 12 + 10]);
    
    if (constraint_residuals[eq] > const_error_thresholds[eq]) {
      all_constraints_satisfied = false;
    }
  }
  
  // Store valid solution
  if (all_constraints_satisfied) {
    unsigned long long storage_index = atomicAdd(&global_result_counter, 1ULL);
    
    if (storage_index < RESULT_CAPACITY) {
      #pragma unroll
      for (int k = 0; k < 10; k++) {
        output_results[storage_index * 10 + k] = position[k];
      }
    }
  }
}

// Comparator for sorting results
int lexicographic_comparator(const void *ptr_a, const void *ptr_b) {
  const double *arr_a = static_cast<const double*>(ptr_a);
  const double *arr_b = static_cast<const double*>(ptr_b);
  
  const double EPSILON = 1e-12;
  
  for (int idx = 0; idx < 10; idx++) {
    double delta = arr_a[idx] - arr_b[idx];
    if (delta < -EPSILON) return -1;
    if (delta > EPSILON) return 1;
  }
  return 0;
}

// Main host function
signed main()
{
  // Host data arrays
  double host_coefficients[120];
  double host_grid_spec[30];

  // Read constraint coefficients from file
  HRTimer start_dataread = HR::now();
  FILE *coeff_file = fopen("problem3-dir/disp.txt", "r");
  if (coeff_file == NULL) {
    cerr << "[ERROR] Could not open disp.txt file" << endl;
    return EXIT_FAILURE;
  }

  int read_count = 0;
  while (!feof(coeff_file)) {
    if (!fscanf(coeff_file, "%lf", &host_coefficients[read_count])) {
      printf("[ERROR]: fscanf failed while reading disp.txt\n");
      exit(EXIT_FAILURE);
    }
    read_count++;
  }
  fclose(coeff_file);

  // Read grid specification from file
  FILE *grid_file = fopen("problem3-dir/grid.txt", "r");
  if (grid_file == NULL) {
    cerr << "[ERROR] Could not open grid.txt file" << endl;
    return EXIT_FAILURE;
  }
  
  int grid_param_count = 0;
  while (!feof(grid_file)) {
    if (!fscanf(grid_file, "%lf", &host_grid_spec[grid_param_count])) {
      printf("[ERROR]: fscanf failed while reading grid.txt\n");
      exit(EXIT_FAILURE);
    }
    grid_param_count++;
  }
  fclose(grid_file);

  HRTimer end_dataread = HR::now();

  // Compute error thresholds
  double tolerance_multiplier = 0.3;
  double host_thresholds[10];
  for (int i = 0; i < 10; i++) {
    host_thresholds[i] = tolerance_multiplier * host_coefficients[12 * i + 11];
  }

  // Compute dimension sizes
  long long host_dim_sizes[10];
  for (int i = 0; i < 10; i++) {
    host_dim_sizes[i] = floor((host_grid_spec[3 * i + 1] - host_grid_spec[3 * i]) / host_grid_spec[3 * i + 2]);
  }

  // Calculate total search space size
  long long total_search_space = 1;
  for (int i = 0; i < 10; i++) {
    if (host_dim_sizes[i] > 0) {
      total_search_space *= host_dim_sizes[i];
    }
  }

  // Allocate host result buffer
  unsigned long long host_result_count = 0;
  double *host_results = static_cast<double*>(malloc(RESULT_CAPACITY * 10 * sizeof(double)));
  if (host_results == NULL) {
    fprintf(stderr, "Error: host result allocation failed\n");
    exit(EXIT_FAILURE);
  }

  // Configure kernel launch parameters
  const int threads_per_block = 1024;

  // Timing events
  cudaEvent_t evt_start, evt_stop, evt_kernel_start, evt_kernel_stop;
  cudaEventCreate(&evt_start);
  cudaEventCreate(&evt_stop);
  cudaEventCreate(&evt_kernel_start);
  cudaEventCreate(&evt_kernel_stop);

  float elapsed_total_ms = 0.0f;
  float elapsed_kernel_ms = 0.0f;

  cudaEventRecord(evt_start);

  // Transfer data to constant memory
  gpuAssert(cudaMemcpyToSymbol(const_grid_params, host_grid_spec, 30 * sizeof(double)), __FILE__, __LINE__);
  gpuAssert(cudaMemcpyToSymbol(const_constraint_matrix, host_coefficients, 120 * sizeof(double)), __FILE__, __LINE__);
  gpuAssert(cudaMemcpyToSymbol(const_tolerance_factor, &tolerance_multiplier, sizeof(double)), __FILE__, __LINE__);
  gpuAssert(cudaMemcpyToSymbol(const_error_thresholds, host_thresholds, 10 * sizeof(double)), __FILE__, __LINE__);
  gpuAssert(cudaMemcpyToSymbol(const_dimension_sizes, host_dim_sizes, 10 * sizeof(long long)), __FILE__, __LINE__);

  // Allocate device output buffer
  double *device_result_buffer = nullptr;
  if (total_search_space > 0) {
    gpuAssert(cudaMalloc(&device_result_buffer, RESULT_CAPACITY * 10 * sizeof(double)), __FILE__, __LINE__);
    gpuAssert(cudaMemset(device_result_buffer, 0, RESULT_CAPACITY * 10 * sizeof(double)), __FILE__, __LINE__);
  }

  cudaEventRecord(evt_kernel_start);

  // Launch kernel if search space is valid
  if (total_search_space > 0) {
    long long iterations_to_process = total_search_space;
    long long blocks_per_grid = (iterations_to_process + threads_per_block - 1) / threads_per_block;
    
    // Reset device counter
    unsigned long long zero_counter = 0;
    gpuAssert(cudaMemcpyToSymbol(global_result_counter, &zero_counter, sizeof(unsigned long long)), __FILE__, __LINE__);

    // Execute kernel
    flattenedGridSearchKernel<<<blocks_per_grid, threads_per_block>>>(
        device_result_buffer, 0, iterations_to_process);
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__);

    // Retrieve result count
    gpuAssert(cudaMemcpyFromSymbol(&host_result_count, global_result_counter, sizeof(unsigned long long)), __FILE__, __LINE__);
    
    if (host_result_count > RESULT_CAPACITY) {
      host_result_count = RESULT_CAPACITY;
    }

    // Copy results back to host
    if (host_result_count > 0) {
      gpuAssert(cudaMemcpy(host_results, device_result_buffer, 
                          host_result_count * 10 * sizeof(double), 
                          cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    }
  }

  cudaEventRecord(evt_kernel_stop);
  cudaEventSynchronize(evt_kernel_stop);

  // Sort results if multiple entries exist
  if (host_result_count > 1) {
    qsort(host_results, host_result_count, 10 * sizeof(double), lexicographic_comparator);
  }

  // Write results to output file
  FILE *output_file = fopen("problem3-dir/results-i.txt", "w");
  if (output_file == NULL) {
    fprintf(stderr, "Error: could not open results-i.txt for writing\n");
    exit(EXIT_FAILURE);
  }

  for (unsigned long long row_idx = 0; row_idx < host_result_count; row_idx++) {
    double *result_entry = host_results + row_idx * 10;
    for (int col_idx = 0; col_idx < 10; col_idx++) {
      fprintf(output_file, "%lf", result_entry[col_idx]);
      if (col_idx < 9) {
        fputc('\t', output_file);
      }
    }
    fputc('\n', output_file);
  }
  fclose(output_file);

  cudaEventRecord(evt_stop);
  cudaEventSynchronize(evt_stop);

  // Calculate timing statistics
  cudaEventElapsedTime(&elapsed_total_ms, evt_start, evt_stop);
  cudaEventElapsedTime(&elapsed_kernel_ms, evt_kernel_start, evt_kernel_stop);

  auto dataread = duration_cast<microseconds>(end_dataread - start_dataread).count();

  cout << "[CPU] Data Read Time: " << dataread << " us" << endl;
  cout << "[GPU] Total Time: " << elapsed_total_ms << " ms" << endl;
  cout << "[GPU] Kernel Compute Time: " << elapsed_kernel_ms << " ms" << endl;
  cout << "Total valid results: " << host_result_count << endl;

  // Cleanup resources
  if (device_result_buffer != nullptr) {
    gpuAssert(cudaFree(device_result_buffer), __FILE__, __LINE__);
  }
  free(host_results);
  cudaEventDestroy(evt_start);
  cudaEventDestroy(evt_stop);
  cudaEventDestroy(evt_kernel_start);
  cudaEventDestroy(evt_kernel_stop);

  return EXIT_SUCCESS;
}