#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <immintrin.h>  // For SSE4 and AVX2 intrinsics
#include <smmintrin.h>  // For SSE4.1
#include <emmintrin.h>   // for streaming stores and _mm_sfence

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

// Tune: i-tile of 2 is usually enough for reuse without thrashing
#ifndef ITILE
#define ITILE 2
#endif

// Original scalar version
void scalar_3d_gradient(const uint64_t* A, uint64_t* B) {
    const uint64_t stride_i = (NY * NZ);
    for (int i = 1; i < NX - 1; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                uint64_t base_idx = (i * NY * NZ) + j * NZ + k;
                // A[i+1, j, k]
                uint64_t A_right = A[base_idx + stride_i];
                // A[i-1, j, k]
                uint64_t A_left = A[base_idx - stride_i];
                B[base_idx] = A_right - A_left;
            }
        }
    }
}

void sse4_3d_gradient(const uint64_t* __restrict A,
                            uint64_t* __restrict B) {
    const uint64_t stride = (uint64_t)NY * NZ;
    const uint64_t bytes_pf_L2 = 3 * 64;
    const uint64_t bytes_pf_L1 = 1 * 64;

    for (int i0 = 1; i0 < (int)NX - 1; i0 += ITILE) {
        const int i_hi = std::min(i0 + ITILE, (int)NX - 1);

        const uint64_t* __restrict pL = A + (uint64_t)(i0 - 1) * stride;
        const uint64_t* __restrict pC = A + (uint64_t)(i0    ) * stride;
        const uint64_t* __restrict pR = A + (uint64_t)(i0 + 1) * stride;

        for (int i = i0; i < i_hi; ++i) {
            uint64_t* __restrict pO = B + (uint64_t)i * stride;

            if (i + 2 < (int)NX) {
                _mm_prefetch((const char*)(A + (uint64_t)(i + 2) * stride), _MM_HINT_T1);
            }

            uint64_t idx = 0;
            const uint64_t vec_lim = stride & ~uint64_t(1);

            for (; idx < vec_lim; idx += 2) {
                _mm_prefetch((const char*)((const char*)(pR + idx) + bytes_pf_L1), _MM_HINT_T0);
                _mm_prefetch((const char*)((const char*)(pL + idx) + bytes_pf_L1), _MM_HINT_T0);

                __m128i vr = _mm_load_si128((const __m128i*)(pR + idx));
                __m128i vl = _mm_load_si128((const __m128i*)(pL + idx));
                __m128i vd = _mm_sub_epi64(vr, vl);

                // Non-temporal store for SSE path
                _mm_stream_si128((__m128i*)(pO + idx), vd);
            }
            for (; idx < stride; ++idx) {
                pO[idx] = pR[idx] - pL[idx];
            }

            pL = pC;
            pC = pR;
            if (i + 2 < (int)NX) {
                pR = A + (uint64_t)(i + 2) * stride;
                _mm_prefetch((const char*)((const char*)pR + bytes_pf_L2), _MM_HINT_T1);
            }
        }
    }

    _mm_sfence();
}


void avx2_3d_gradient(const uint64_t* __restrict A,
                            uint64_t* __restrict B) {
    const uint64_t stride = (uint64_t)NY * NZ; // elements per plane
    const uint64_t bytes_pf_L2 = 3 * 64;       // ~3 cache lines lookahead to L2
    const uint64_t bytes_pf_L1 = 1 * 64;       // ~1 cache line lookahead to L1

    // Process i in tiles, each tile needs planes [i0-1 .. i0+ITILE]
    for (int i0 = 1; i0 < (int)NX - 1; i0 += ITILE) {
        const int i_hi = std::min(i0 + ITILE, (int)NX - 1);

        // Seed the 3-plane window: L = i0-1, C = i0, R = i0+1
        const uint64_t* __restrict pL = A + (uint64_t)(i0 - 1) * stride;
        const uint64_t* __restrict pC = A + (uint64_t)(i0    ) * stride;
        const uint64_t* __restrict pR = A + (uint64_t)(i0 + 1) * stride;

        for (int i = i0; i < i_hi; ++i) {
            uint64_t* __restrict pO = B + (uint64_t)i * stride;

            // Prefetch the future right plane for the next i (i+2) into L2
            if (i + 2 < (int)NX) {
                _mm_prefetch((const char*)(A + (uint64_t)(i + 2) * stride), _MM_HINT_T1);
            }

            // Linear pass over the plane: 4x u64 per vector
            uint64_t idx = 0;
            const uint64_t vec_lim = stride & ~uint64_t(3);

            for (; idx < vec_lim; idx += 4) {
                // Within-plane L1 prefetch for both sources
                _mm_prefetch((const char*)((const char*)(pR + idx) + bytes_pf_L1), _MM_HINT_T0);
                _mm_prefetch((const char*)((const char*)(pL + idx) + bytes_pf_L1), _MM_HINT_T0);

                __m256i vr = _mm256_load_si256((const __m256i*)(pR + idx));
                __m256i vl = _mm256_load_si256((const __m256i*)(pL + idx));
                __m256i vd = _mm256_sub_epi64(vr, vl);

                // Non-temporal (streaming) store: avoid write-allocate on B
                _mm256_stream_si256((__m256i*)(pO + idx), vd);
            }

            // Scalar tail (<=3)
            for (; idx < stride; ++idx) {
                pO[idx] = pR[idx] - pL[idx];
            }

            // Rotate planes for next i:
            // new L = old C, new C = old R, new R = A[i+2]
            pL = pC;
            pC = pR;
            if (i + 2 < (int)NX) {
                pR = A + (uint64_t)(i + 2) * stride;

                // L2 prefetch a bit deeper for the new right plane
                _mm_prefetch((const char*)((const char*)pR + bytes_pf_L2), _MM_HINT_T1);
            }
        }
    }

    // Ensure all NT stores are globally visible before timing stops or verification
    _mm_sfence();
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

// Function to verify results match
bool verify_results(const uint64_t* grid1, const uint64_t* grid2, const char* name) {
    for (int i = 0; i < TOTAL_SIZE; i++) {
        if (grid1[i] != grid2[i]) {
            cout << "ERROR: " << name << " result mismatch at index " << i 
                 << " (expected: " << grid1[i] << ", got: " << grid2[i] << ")" << endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Allocate aligned memory for better SIMD performance
    auto* i_grid = (uint64_t*)aligned_alloc(32, TOTAL_SIZE * sizeof(uint64_t));
    
    // Initialize input grid
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                i_grid[i*NY*NZ+j*NZ+k] = (INITIAL_VAL + i + 2 * j + 3 * k);
            }
        }
    }
    
    // Allocate output grids
    auto* o_grid_scalar = (uint64_t*)aligned_alloc(32, TOTAL_SIZE * sizeof(uint64_t));
    auto* o_grid_sse4 = (uint64_t*)aligned_alloc(32, TOTAL_SIZE * sizeof(uint64_t));
    auto* o_grid_avx2 = (uint64_t*)aligned_alloc(32, TOTAL_SIZE * sizeof(uint64_t));
    
    // Initialize output grids
    std::fill_n(o_grid_scalar, TOTAL_SIZE, 0);
    std::fill_n(o_grid_sse4, TOTAL_SIZE, 0);
    std::fill_n(o_grid_avx2, TOTAL_SIZE, 0);
    
    cout << "Running 3D Gradient Kernel Performance Comparison" << endl;
    cout << "Grid size: " << NX << "x" << NY << "x" << NZ << endl;
    cout << "Iterations: " << N_ITERATIONS << endl;
    cout << "================================================" << endl << endl;
    
    // Scalar version timing
    auto start = HR::now();
    for (int iter = 0; iter < N_ITERATIONS; ++iter) {
        scalar_3d_gradient(i_grid, o_grid_scalar);
    }
    auto end = HR::now();
    auto scalar_duration = duration_cast<milliseconds>(end - start).count();
    cout << "Scalar kernel time (ms): " << scalar_duration << endl;
    
    // SSE4 version timing
    start = HR::now();
    for (int iter = 0; iter < N_ITERATIONS; ++iter) {
        sse4_3d_gradient(i_grid, o_grid_sse4);
    }
    end = HR::now();
    auto sse4_duration = duration_cast<milliseconds>(end - start).count();
    cout << "SSE4 kernel time (ms):   " << sse4_duration << endl;
    
    // AVX2 version timing
    start = HR::now();
    for (int iter = 0; iter < N_ITERATIONS; ++iter) {
        avx2_3d_gradient(i_grid, o_grid_avx2);
    }
    end = HR::now();
    auto avx2_duration = duration_cast<milliseconds>(end - start).count();
    cout << "AVX2 kernel time (ms):   " << avx2_duration << endl;
    
    cout << endl << "Performance Analysis" << endl;
    cout << "====================" << endl;
    
    // Compute and verify checksums
    uint64_t scalar_checksum = compute_checksum(o_grid_scalar);
    uint64_t sse4_checksum = compute_checksum(o_grid_sse4);
    uint64_t avx2_checksum = compute_checksum(o_grid_avx2);
    
    cout << "Scalar checksum: " << scalar_checksum << endl;
    cout << "SSE4 checksum:   " << sse4_checksum << endl;
    cout << "AVX2 checksum:   " << avx2_checksum << endl;
    
    // Verify results
    bool sse4_correct = verify_results(o_grid_scalar, o_grid_sse4, "SSE4");
    bool avx2_correct = verify_results(o_grid_scalar, o_grid_avx2, "AVX2");
    
    if (sse4_correct && avx2_correct) {
        cout << endl << "✓ All implementations produce identical results" << endl;
    }
    
    // Calculate speedups
    cout << endl << "Speedup Analysis" << endl;
    cout << "================" << endl;
    
    double sse4_speedup = (double)scalar_duration / sse4_duration;
    double avx2_speedup = (double)scalar_duration / avx2_duration;
    
    cout << std::fixed << std::setprecision(2);
    cout << "SSE4 speedup over scalar:  " << sse4_speedup << "x" << endl;
    cout << "AVX2 speedup over scalar:  " << avx2_speedup << "x" << endl;
    cout << "AVX2 speedup over SSE4:    " << (double)sse4_duration / avx2_duration << "x" << endl;
    
    // Performance metrics
    cout << endl << "Performance Metrics" << endl;
    cout << "==================" << endl;
    
    uint64_t num_operations = (uint64_t)(NX - 2) * NY * NZ * N_ITERATIONS;
    double scalar_gops = (double)num_operations / (scalar_duration * 1e6);
    double sse4_gops = (double)num_operations / (sse4_duration * 1e6);
    double avx2_gops = (double)num_operations / (avx2_duration * 1e6);
    
    cout << "Scalar GOPS: " << scalar_gops << endl;
    cout << "SSE4 GOPS:   " << sse4_gops << endl;
    cout << "AVX2 GOPS:   " << avx2_gops << endl;
    
    // Memory bandwidth estimation
    uint64_t bytes_accessed = (uint64_t)(NX - 2) * NY * NZ * 2 * sizeof(uint64_t) * N_ITERATIONS; // 2 reads per element
    double scalar_bandwidth = (double)bytes_accessed / (scalar_duration * 1e6); // GB/s
    double sse4_bandwidth = (double)bytes_accessed / (sse4_duration * 1e6);
    double avx2_bandwidth = (double)bytes_accessed / (avx2_duration * 1e6);
    
    cout << endl << "Memory Bandwidth (GB/s)" << endl;
    cout << "======================" << endl;
    cout << "Scalar: " << scalar_bandwidth << endl;
    cout << "SSE4:   " << sse4_bandwidth << endl;
    cout << "AVX2:   " << avx2_bandwidth << endl;


    cout << "Scalar              & \\texttt{" << scalar_duration << "} & $1.00\\times$ & \\texttt{" << scalar_gops << "} & \\texttt{" << scalar_bandwidth << "}\\\\" << endl;
    cout << "SSE4  & \\texttt{" << sse4_duration << "} & \\texttt{" << sse4_speedup << "}$\\times$ & \\texttt{" << sse4_gops << "} & \\texttt{" << sse4_bandwidth << "}\\\\" << endl;
    cout << "AVX2   & \\texttt{" << avx2_duration << "} & \\texttt{" << avx2_speedup << "}$\\times$ & \\texttt{" << avx2_gops << "} & \\texttt{" << avx2_bandwidth << "}\\\\" << endl;

    
    // Cleanup
    free(i_grid);
    free(o_grid_scalar);
    free(o_grid_sse4);
    free(o_grid_avx2);
    
    return EXIT_SUCCESS;
}