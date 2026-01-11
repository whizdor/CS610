#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
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

// Functor to convert linear index to 10D point and check constraints
struct GridPointValidator {
    double grid[30];
    double coeffs[120];
    double kk;
    int bounds[10];
    unsigned long long total_iterations;
    
    __host__ __device__
    GridPointValidator(const double* h_grid, const double* h_coeffs, double k) : kk(k) {
        // Copy data to local arrays
        for (int i = 0; i < 30; i++) grid[i] = h_grid[i];
        for (int i = 0; i < 120; i++) coeffs[i] = h_coeffs[i];
        
        // Precompute bounds
        total_iterations = 1;
        for (int i = 0; i < 10; i++) {
            bounds[i] = floor((grid[i*3+1] - grid[i*3]) / grid[i*3+2]);
            total_iterations *= bounds[i];
        }
    }
    
    __host__ __device__
    bool operator()(unsigned long long idx) const {
        if (idx >= total_iterations) return false;
        
        // Convert linear index to 10D coordinates
        int indices[10];
        unsigned long long temp = idx;
        for (int i = 9; i >= 0; i--) {
            indices[i] = temp % bounds[i];
            temp /= bounds[i];
        }
        
        // Calculate x values
        double x[10];
        for (int i = 0; i < 10; i++) {
            x[i] = grid[i*3] + indices[i] * grid[i*3+2];
        }
        
        // Check constraints
        for (int i = 0; i < 10; i++) {
            double sum = 0.0;
            for (int j = 0; j < 10; j++) {
                sum += coeffs[i*12+j] * x[j];
            }
            double q = fabs(sum - coeffs[i*12+10]);
            double threshold = kk * coeffs[i*12+11];
            if (q > threshold) return false;
        }
        
        return true;
    }
};

// Functor to transform valid indices to 10D points
struct IndexToPoint {
    double grid[30];
    int bounds[10];
    
    __host__ __device__
    IndexToPoint(const double* h_grid) {
        for (int i = 0; i < 30; i++) grid[i] = h_grid[i];
        for (int i = 0; i < 10; i++) {
            bounds[i] = floor((grid[i*3+1] - grid[i*3]) / grid[i*3+2]);
        }
    }
    
    __host__ __device__
    void operator()(unsigned long long idx, double* x) const {
        int indices[10];
        unsigned long long temp = idx;
        for (int i = 9; i >= 0; i--) {
            indices[i] = temp % bounds[i];
            temp /= bounds[i];
        }
        
        for (int i = 0; i < 10; i++) {
            x[i] = grid[i*3] + indices[i] * grid[i*3+2];
        }
    }
    
    __host__ __device__
    void getIndices(unsigned long long idx, int* indices) const {
        unsigned long long temp = idx;
        for (int i = 9; i >= 0; i--) {
            indices[i] = temp % bounds[i];
            temp /= bounds[i];
        }
    }
};

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
  
  // Calculate total iteration space
  unsigned long long total_iterations = 1;
  for (int i = 0; i < 10; i++) {
      int s = floor((h_grid[i*3+1] - h_grid[i*3]) / h_grid[i*3+2]);
      total_iterations *= s;
  }
  
  HRTimer start_memory = HR::now();
  
  // Create device vectors
  thrust::device_vector<double> d_grid(h_grid, h_grid + 30);
  thrust::device_vector<double> d_coeffs(h_coeffs, h_coeffs + 120);
  
  // Create validator functor
  GridPointValidator validator(h_grid, h_coeffs, kk);
  
  HRTimer start_compute = HR::now();
  
  // Method 1: Using counting iterator and copy_if
  thrust::counting_iterator<unsigned long long> first(0);
  thrust::counting_iterator<unsigned long long> last(total_iterations);
  
  // Count valid points
  int num_valid = thrust::count_if(first, last, validator);
  cout << "Thrust found " << num_valid << " valid points\n";
  
  // Allocate space for valid indices
  thrust::device_vector<unsigned long long> valid_indices(num_valid);
  
  // Copy valid indices
  thrust::copy_if(first, last, valid_indices.begin(), validator);
  
  cudaDeviceSynchronize();
  HRTimer end_compute = HR::now();
  
  // Transform indices to points
  IndexToPoint transformer(h_grid);
  
  // Copy results back to host
  thrust::host_vector<unsigned long long> h_valid_indices = valid_indices;
  
  // Create pairs of (index, result) for sorting
  std::vector<std::pair<std::array<int, 10>, std::array<double, 10>>> pairs;
  
  for (int i = 0; i < num_valid; i++) {
      std::array<int, 10> idx_arr;
      std::array<double, 10> res_arr;
      
      double x[10];
      int indices[10];
      transformer.getIndices(h_valid_indices[i], indices);
      transformer(h_valid_indices[i], x);
      
      for (int j = 0; j < 10; j++) {
          idx_arr[j] = indices[j];
          res_arr[j] = x[j];
      }
      pairs.push_back({idx_arr, res_arr});
  }
  
  // Sort by indices (lexicographically)
  std::sort(pairs.begin(), pairs.end(), 
    [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
  
  // Write sorted results to file
  fp = fopen("problem3-dir/results-iv.txt", "w");
  for (const auto& p : pairs)
  {
    for (int j = 0; j < 10; j++)
    {
      fprintf(fp, "%lf\t", p.second[j]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

  auto dataread = duration_cast<microseconds>(end_dataread - start_dataread).count();
  auto memory = duration_cast<milliseconds>(start_compute - start_memory).count();
  auto compute = duration_cast<milliseconds>(end_compute - start_compute).count();
  auto total = duration_cast<milliseconds>(end_compute - start_dataread).count();
  cout << "[CPU] Data Read Time: " << dataread << " us" << endl;
  cout << "[GPU] Memory HtoD Time: " << memory << " ms" << endl;
  cout << "[GPU] Kernel Compute Time: " << compute << " ms" << endl;
  cout << "[GPU] Total Time: " << total << " ms" << endl;

  return EXIT_SUCCESS;
}
