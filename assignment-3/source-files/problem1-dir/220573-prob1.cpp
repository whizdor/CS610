#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits.h>
#include <omp.h>
#include <immintrin.h>
#include <cassert>

using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const uint64_t TIMESTEPS = 100;

const double W_OWN = (1.0 / 7.0);
const double W_NEIGHBORS = (1.0 / 7.0);

const uint64_t NX = 258; // 64 interior points + 2 boundary points
const uint64_t NY = 258;
const uint64_t NZ = 258;
const uint64_t TOTAL_SIZE = NX * NY * NZ;

const static double EPSILON = 1e-10;

// Original baseline kernel
void stencil_3d_7pt_baseline(const double* curr, double* next) {
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];

        next[i * NY * NZ + j * NZ + k] = W_OWN * curr[i * NY * NZ + j * NZ + k] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Version 1: OpenMP Parallelization using collapse(2)
void stencil_3d_7pt_omp_collapse(const double* curr, double* next) {
  #pragma omp parallel for collapse(2)
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];

        next[i * NY * NZ + j * NZ + k] = W_OWN * curr[i * NY * NZ + j * NZ + k] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Version 2: OpemMP Parallelization using for schedule(static) 
void stencil_3d_7pt_omp_static(const double* curr, double* next) {
  #pragma omp parallel for schedule(static)
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];

        next[i * NY * NZ + j * NZ + k] = W_OWN * curr[i * NY * NZ + j * NZ + k] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Version 3: Loop Unrolling (2x in k dimension)
void stencil_3d_7pt_unrolled(const double* curr, double* next) {
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      int k;
      for (k = 1; k < NZ - 2; k += 2) {
        // First iteration
        double neighbors_sum1 = 0.0;
        neighbors_sum1 += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum1 += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum1 += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum1 += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum1 += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum1 += curr[i * NY * NZ + j * NZ + (k - 1)];

        // Second iteration
        double neighbors_sum2 = 0.0;
        neighbors_sum2 += curr[(i + 1) * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum2 += curr[(i - 1) * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum2 += curr[i * NY * NZ + (j + 1) * NZ + (k + 1)];
        neighbors_sum2 += curr[i * NY * NZ + (j - 1) * NZ + (k + 1)];
        neighbors_sum2 += curr[i * NY * NZ + j * NZ + (k + 2)];
        neighbors_sum2 += curr[i * NY * NZ + j * NZ + k];

        next[i * NY * NZ + j * NZ + k] = W_OWN * curr[i * NY * NZ + j * NZ + k] + W_NEIGHBORS * neighbors_sum1;
        next[i * NY * NZ + j * NZ + (k + 1)] = W_OWN * curr[i * NY * NZ + j * NZ + (k + 1)] + W_NEIGHBORS * neighbors_sum2;
      }
      // Handle remainder
      for (; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];

        next[i * NY * NZ + j * NZ + k] = W_OWN * curr[i * NY * NZ + j * NZ + k] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Version 4: Tiling
void stencil_3d_7pt_tiled(const double* curr, double* next) {
  const int TILE_I = 16;
  const int TILE_J = 16;
  const int TILE_K = 16;

  for (int ii = 1; ii < NX - 1; ii += TILE_I) {
    for (int jj = 1; jj < NY - 1; jj += TILE_J) {
      for (int kk = 1; kk < NZ - 1; kk += TILE_K) {
        for (int i = ii; i < std::min(ii + TILE_I, (int)(NX - 1)); ++i) {
          for (int j = jj; j < std::min(jj + TILE_J, (int)(NY - 1)); ++j) {
            for (int k = kk; k < std::min(kk + TILE_K, (int)(NZ - 1)); ++k) {
              double neighbors_sum = 0.0;
              neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
              neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
              neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
              neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
              neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
              neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];

              next[i * NY * NZ + j * NZ + k] = W_OWN * curr[i * NY * NZ + j * NZ + k] + W_NEIGHBORS * neighbors_sum;
            }
          }
        }
      }
    }
  }
}

// Version 5: Loop unrolling (2x in k dimension) with OpenMP
void stencil_3d_7pt_unrolled_omp(const double* curr, double* next) {
  #pragma omp parallel for collapse(2)
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      int k;
      for (k = 1; k < NZ - 2; k += 2) {
        // First iteration
        double neighbors_sum1 = 0.0;
        neighbors_sum1 += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum1 += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum1 += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum1 += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum1 += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum1 += curr[i * NY * NZ + j * NZ + (k - 1)];
        
        // Second iteration
        double neighbors_sum2 = 0.0;
        neighbors_sum2 += curr[(i + 1) * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum2 += curr[(i - 1) * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum2 += curr[i * NY * NZ + (j + 1) * NZ + (k + 1)];
        neighbors_sum2 += curr[i * NY * NZ + (j - 1) * NZ + (k + 1)];
        neighbors_sum2 += curr[i * NY * NZ + j * NZ + (k + 2)];
        neighbors_sum2 += curr[i * NY * NZ + j * NZ + k];

        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum1;
            
        next[i * NY * NZ + j * NZ + (k + 1)] =
            W_OWN * curr[i * NY * NZ + j * NZ + (k + 1)] +
            W_NEIGHBORS * neighbors_sum2;
      }
      
      // Handle remainder
      for (; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];

        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Version 6: OpenMP + SIMD
void stencil_3d_7pt_omp_simd(const double* curr, double* next) {
  const int plane = NY * NZ;
  const int row   = NZ;

  // Single parallel region, then workshare. This avoids repeated fork/join.
  #pragma omp parallel
  {
    #pragma omp for collapse(2) schedule(static)
    for (int i = 1; i < (int)NX - 1; ++i) {
      for (int j = 1; j < (int)NY - 1; ++j) {
        const int base = i * plane + j * row;

        // Prefetch some rows ahead to help hardware prefetchers a bit
        __builtin_prefetch(&curr[base + 32], 0, 3);
        int k;
        // Vectorize the innermost loop over k.
        #pragma omp simd simdlen(4) safelen(4) linear(k:1) aligned(curr,next:64)
        for (k = 1; k < (int)NZ - 1; ++k) {
          const int idx = base + k;

          const double c_ip = curr[idx + plane];
          const double c_im = curr[idx - plane];
          const double c_jp = curr[idx + row];
          const double c_jm = curr[idx - row];
          const double c_kp = curr[idx + 1];
          const double c_km = curr[idx - 1];

          const double nsum = c_ip + c_im + c_jp + c_jm + c_kp + c_km;
          next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * nsum;
        }
      }
    }
  }
}

// Version 7: Best combination - Tiled + Unrolled + OpenMP + SIMD
void stencil_3d_7pt_best(const double* curr, double* next) {
  const int TILE_I = 16;
  const int TILE_J = 16; 
  const int TILE_K = 32;
  
  const int plane = NY * NZ;
  const int row = NZ;

  #pragma omp parallel
  {
    #pragma omp for collapse(2) schedule(static)
    for (int ii = 1; ii < (int)NX - 1; ii += TILE_I) {
      for (int jj = 1; jj < (int)NY - 1; jj += TILE_J) {
        for (int kk = 1; kk < (int)NZ - 1; kk += TILE_K) {
          // Process tiles
          for (int i = ii; i < std::min(ii + TILE_I, (int)(NX - 1)); ++i) {
            for (int j = jj; j < std::min(jj + TILE_J, (int)(NY - 1)); ++j) {
              const int base = i * plane + j * row;
              
              // Prefetch next tile data
              if (i + 1 < NX - 1) {
                __builtin_prefetch(&curr[base + plane + 32], 0, 3);
              }
              int k;
              int k_end = std::min(kk + TILE_K, (int)(NZ - 1));
              
              // Unroll by 2 with SIMD hints
              #pragma omp simd simdlen(2) safelen(2) aligned(curr,next:64)
  
              for (k = std::max(kk, 1); k < k_end - 1; k += 2) {
                // First point
                const int idx1 = base + k;
                const double c_ip1 = curr[idx1 + plane];
                const double c_im1 = curr[idx1 - plane];
                const double c_jp1 = curr[idx1 + row];
                const double c_jm1 = curr[idx1 - row];
                const double c_kp1 = curr[idx1 + 1];
                const double c_km1 = curr[idx1 - 1];
                
                const double nsum1 = c_ip1 + c_im1 + c_jp1 + c_jm1 + c_kp1 + c_km1;
                next[idx1] = W_OWN * curr[idx1] + W_NEIGHBORS * nsum1;
                
                // Second point
                const int idx2 = base + k + 1;
                const double c_ip2 = curr[idx2 + plane];
                const double c_im2 = curr[idx2 - plane];
                const double c_jp2 = curr[idx2 + row];
                const double c_jm2 = curr[idx2 - row];
                const double c_kp2 = curr[idx2 + 1];
                const double c_km2 = curr[idx2 - 1];
                
                const double nsum2 = c_ip2 + c_im2 + c_jp2 + c_jm2 + c_kp2 + c_km2;
                next[idx2] = W_OWN * curr[idx2] + W_NEIGHBORS * nsum2;
              }
              
              // Handle remainder
              for (; k < k_end; ++k) {
                const int idx = base + k;
                const double c_ip = curr[idx + plane];
                const double c_im = curr[idx - plane];
                const double c_jp = curr[idx + row];
                const double c_jm = curr[idx - row];
                const double c_kp = curr[idx + 1];
                const double c_km = curr[idx - 1];
                
                const double nsum = c_ip + c_im + c_jp + c_jm + c_kp + c_km;
                next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * nsum;
              }
            }
          }
        }
      }
    }
  }
}

// Helper function to run a kernel and measure its performance
void run_kernel(void (*kernel)(const double*, double*), const char* name,
                double expected_final, double expected_sum) {
  auto* grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  auto* grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  double* current_grid = grid1;
  double* next_grid = grid2;

  // Warmup
  for (int t = 0; t < 5; t++) {
    kernel(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  
  // Reset grids
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;
  current_grid = grid1;
  next_grid = grid2;

  auto start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++) {
    kernel(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  auto end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  
  double final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  double total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++) {
    total_sum += current_grid[i];
  }
  
  cout << name << " time: " << duration << " ms" << endl;
  cout << "  Final value at center: " << final << endl;
  cout << "  Total sum: " << total_sum << endl;
  
  // Verify correctness
  assert(std::abs(final - expected_final) < EPSILON && "Final value mismatch!");
  assert(std::abs(total_sum - expected_sum) < EPSILON && "Total sum mismatch!");

  delete[] grid1;
  delete[] grid2;
}

int main() {
  cout << "=== 3D Stencil Performance Comparison ===" << endl;
  cout << "Grid size: " << NX << "x" << NY << "x" << NZ << endl;
  cout << "Timesteps: " << TIMESTEPS << endl;
  cout << "Threads: " << omp_get_max_threads() << endl << endl;

  // Run baseline first to get reference values
  auto* grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  auto* grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  double* current_grid = grid1;
  double* next_grid = grid2;

  auto start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++) {
    stencil_3d_7pt_baseline(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  auto end = HR::now();
  auto baseline_time = duration_cast<milliseconds>(end - start).count();
  
  double expected_final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  double expected_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++) {
    expected_sum += current_grid[i];
  }
  
  cout << "Baseline (scalar) kernel time: " << baseline_time << " ms" << endl;
  cout << "  Final value at center: " << expected_final << endl;
  cout << "  Total sum: " << expected_sum << endl << endl;

  delete[] grid1;
  delete[] grid2;

  // Run all optimized versions
  run_kernel(stencil_3d_7pt_omp_collapse, "OpenMP Collapse", expected_final, expected_sum);
  run_kernel(stencil_3d_7pt_omp_static, "OpenMP Static", expected_final, expected_sum);
  run_kernel(stencil_3d_7pt_unrolled, "Unrolled only", expected_final, expected_sum);
  run_kernel(stencil_3d_7pt_tiled, "Tiled only", expected_final, expected_sum);
  run_kernel(stencil_3d_7pt_unrolled_omp, "Unrolled + OpenMP", expected_final, expected_sum);
  run_kernel(stencil_3d_7pt_omp_simd, "OpenMP + SIMD", expected_final, expected_sum);
  run_kernel(stencil_3d_7pt_best, "Best (Tiled + Unrolled + OpenMP)", expected_final, expected_sum);
  return EXIT_SUCCESS;
}