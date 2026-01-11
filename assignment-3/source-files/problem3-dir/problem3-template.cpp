#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits.h>

using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const uint32_t NX = 128;
const uint32_t NY = 128;
const uint32_t NZ = 128;
const uint64_t TOTAL_SIZE = (NX * NY * NZ);

const uint32_t N_ITERATIONS = 100;
const uint64_t INITIAL_VAL = 1000000;

void scalar_3d_gradient(const uint64_t* A, uint64_t* B) {
  const uint64_t stride_i = (NY * NZ);
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 0; j < NY; ++j) {
      for (int k = 0; k < NZ; ++k) {
        uint64_t base_idx = (i * NY * NZ) + j * NZ + k;
        // A[i+1, j, k]
        int A_right = A[base_idx + stride_i];
        // A[i-1, j, k]
        int A_left = A[base_idx - stride_i];
        B[base_idx] = A_right - A_left;
      }
    }
  }
}

long compute_checksum(const uint64_t* grid) {
  uint64_t sum = 0;
  for (int i = 1; i < (NX - 1); i++) {
    for (int j = 0; j < NY; j++) {
      for (int k = 0; k < NZ; k++) {
        sum += grid[i * NY * NZ + j * NZ + k];
      }
    }
  }
  return sum;
}

int main() {
  auto* i_grid = new uint64_t[TOTAL_SIZE];
   for (int i = 0; i < NX; i++) {
    for (int j = 0; j < NY; j++) {
      for (int k = 0; k < NZ; k++) {
        i_grid[i*NY*NZ+j*NZ+k] = (INITIAL_VAL + i +
                                  2 * j + 3 * k);
      }
    }
  }

  auto* o_grid1 = new uint64_t[TOTAL_SIZE];
  std::fill_n(o_grid1, TOTAL_SIZE, 0);

  auto start = HR::now();
  for (int iter = 0; iter < N_ITERATIONS; ++iter) {
    scalar_3d_gradient(i_grid, o_grid1);
  }
  auto end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << "Scalar kernel time (ms): " << duration << "\n";

  // Compare checksum with vector versions
  uint64_t scalar_checksum = compute_checksum(o_grid1);
  cout << "Checksum: " << scalar_checksum << "\n";

  // Assert the checksum for vectors variants

  delete[] i_grid;
  delete[] o_grid1;

  return EXIT_SUCCESS;
}
