#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <x86intrin.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

#define N (1 << 24)
#define SSE_WIDTH_BITS (128)
#define AVX2_WIDTH_BITS (256)
#define ALIGN (32)

/** Helper methods for debugging */

// Print all elements of the array
void print_array(const int *array)
{
  for (int i = 0; i < N; i++)
  {
    cout << array[i] << "\t";
  }
  cout << "\n";
}

// Print 128-bit vector as 4 unsigned 32-bit integers
void print128i_u32(__m128i var, int start)
{
  alignas(ALIGN) uint32_t val[4];
  _mm_store_si128(reinterpret_cast<__m128i *>(val), var);
  cout << "Values [" << start << ":" << start + 3 << "]: " << val[0] << " "
       << val[1] << " " << val[2] << " " << val[3] << "\n";
}

// Print 128-bit vector as 2 unsigned 64-bit integers
void print128i_u64(__m128i var)
{
  alignas(ALIGN) uint64_t val[2];
  _mm_store_si128(reinterpret_cast<__m128i *>(val), var);
  cout << "Values [0:1]: " << val[0] << " " << val[1] << "\n";
}

// Print 256-bit vector as 8 unsigned 32-bit integers
void print256i_u32(__m256i var, int start)
{
  alignas(ALIGN) uint32_t val[8];
  _mm256_store_si256(reinterpret_cast<__m256i *>(val), var);
  cout << "Values [" << start << ":" << start + 7 << "]: ";
  for (int i = 0; i < 8; i++)
  {
    cout << val[i] << " ";
  }
  cout << "\n";
}

// Reference serial implementation of prefix sum
__attribute__((optimize("no-tree-vectorize"))) int
ref_version(int *__restrict__ source, int *__restrict__ dest)
{
  source = static_cast<int *>(__builtin_assume_aligned(source, ALIGN));
  dest = static_cast<int *>(__builtin_assume_aligned(dest, ALIGN));

  int tmp = 0;
  for (int i = 0; i < N; i++)
  {
    tmp += source[i];
    dest[i] = tmp;
  }
  return tmp;
}

// OpenMP SIMD implementation with scan directive
int omp_version(const int *__restrict__ source, int *__restrict__ dest)
{
  source = static_cast<int *>(__builtin_assume_aligned(source, ALIGN));
  dest = static_cast<int *>(__builtin_assume_aligned(dest, ALIGN));

  int tmp = 0;
#pragma omp simd reduction(inscan, + : tmp)
  for (int i = 0; i < N; i++)
  {
    tmp += source[i];
#pragma omp scan inclusive(tmp)
    dest[i] = tmp;
  }
  return tmp;
}

// SSE4 implementation using tree reduction on 128-bit vectors
int sse4_version(const int *__restrict__ source, int *__restrict__ dest)
{
  source = static_cast<int *>(__builtin_assume_aligned(source, ALIGN));
  dest = static_cast<int *>(__builtin_assume_aligned(dest, ALIGN));

  // Initialize offset vector to zero for first iteration
  __m128i offset = _mm_setzero_si128();

  const int stride = SSE_WIDTH_BITS / (sizeof(int) * CHAR_BIT);
  for (int i = 0; i < N; i += stride)
  {
    // Load 4 integers from aligned memory
    __m128i x = _mm_load_si128((__m128i *)&source[i]); // [a b c d]
    // First reduction step: shift left by 1 position and add
    __m128i tmp0 = _mm_slli_si128(x, 4); // [b c d 0]
    __m128i tmp1 = _mm_add_epi32(x, tmp0); // [a+b b+c c+d d]
    // Second reduction step: shift left by 2 positions and add
    __m128i tmp2 = _mm_slli_si128(tmp1, 8); // [c+d d 0 0]
    // Complete tree reduction for this block
    __m128i out = _mm_add_epi32(tmp2, tmp1); // [a+b+c+d b+c+d c+d d]
    // Add offset from previous block
    out = _mm_add_epi32(out, offset); 

    // Store result to aligned memory
    _mm_store_si128(reinterpret_cast<__m128i *>(&dest[i]), out);

    // Broadcast the last element (cumulative sum) to all positions
    // This becomes the offset for the next iteration
    offset = _mm_shuffle_epi32(out, _MM_SHUFFLE(3, 3, 3, 3));
  }
  return dest[N - 1];
}

// AVX2 implementation using tree reduction on 256-bit vectors
int avx2_version(const int *source, int *dest)
{
  source = static_cast<const int *>(__builtin_assume_aligned(source, ALIGN));
  dest = static_cast<int *>(__builtin_assume_aligned(dest, ALIGN));

  // Initialize offset vector to zero for first iteration
  __m256i offset = _mm256_setzero_si256();

  const int stride = AVX2_WIDTH_BITS / (sizeof(int) * CHAR_BIT);

  for (int i = 0; i < N; i += stride)
  {
    // Load 8 integers from aligned memory
    __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i *>(&source[i]));

    // First reduction step: shift left by 1 position and add
    __m256i tmp1 = _mm256_slli_si256(x, 4);
    __m256i sum1 = _mm256_add_epi32(x, tmp1);

    // Second reduction step: shift left by 2 positions and add
    __m256i tmp2 = _mm256_slli_si256(sum1, 8);
    __m256i sum2 = _mm256_add_epi32(sum1, tmp2);

    // Handle cross-lane boundary for complete tree reduction
    __m128i high = _mm256_extracti128_si256(sum2, 1);
    __m128i low_last = _mm_shuffle_epi32(_mm256_castsi256_si128(sum2), _MM_SHUFFLE(3, 3, 3, 3));
    high = _mm_add_epi32(high, low_last);

    // Reconstruct the complete result
    __m256i out = _mm256_inserti128_si256(sum2, high, 1);

    // Add offset from previous block
    out = _mm256_add_epi32(out, offset);

    // Store result to aligned memory
    _mm256_store_si256(reinterpret_cast<__m256i *>(&dest[i]), out);

    // Extract and broadcast the last element for next iteration
    __m128i out_high = _mm256_extracti128_si256(out, 1);
    offset = _mm256_set1_epi32(_mm_extract_epi32(out_high, 3));
  }

  return dest[N - 1];
}

// Main function to benchmark different prefix sum implementations
__attribute__((optimize("no-tree-vectorize"))) int main()
{
  int *array = static_cast<int *>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(array, array + N, 1);

  int *ref_res = static_cast<int *>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(ref_res, ref_res + N, 0);
  HRTimer start = HR::now();
  int val_ser = ref_version(array, ref_res);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial version: " << val_ser << " time: " << duration << endl;

  int *omp_res = static_cast<int *>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(omp_res, omp_res + N, 0);
  start = HR::now();
  int val_omp = omp_version(array, omp_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_omp || printf("OMP result is wrong!\n"));
  cout << "OMP version: " << val_omp << " time: " << duration << endl;

  int *sse_res = static_cast<int *>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(sse_res, sse_res + N, 0);
  start = HR::now();
  int val_sse = sse4_version(array, sse_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_sse || printf("SSE result is wrong!\n"));
  cout << "SSE version: " << val_sse << " time: " << duration << endl;

  int *avx2_res = static_cast<int *>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(avx2_res, avx2_res + N, 0);
  start = HR::now();
  int val_avx2 = avx2_version(array, avx2_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_avx2 || printf("AVX2 result is wrong!\n"));
  cout << "AVX2 version: " << val_avx2 << " time: " << duration << endl;

  // Clean up allocated memory
  free(array);
  free(ref_res);
  free(omp_res);
  free(sse_res);
  free(avx2_res);

  return EXIT_SUCCESS;
}
