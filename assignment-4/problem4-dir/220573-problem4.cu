#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <limits>
#include <sys/time.h>
#include <cmath>

#define THRESHOLD (std::numeric_limits<float>::epsilon())

using std::cerr;
using std::cout;
using std::endl;

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

// Problem Size
const uint64_t N = (1 << 6);
const int TILE_SIZE = 4;
const int filterSize2D = 3;
const int filterRadius2D = filterSize2D >> 1;
const int filterSize3D = 3;
const int filterRadius3D = filterSize3D >> 1;

// 2D Convolution - CPU version
__host__ void kernel2D_cpu(const float *input, float *output,
                           const float *filter, int width, int height,
                           int filterSize)
{
  const int radius = filterSize >> 1;

  for (int row = 0; row < height; ++row)
  {
    for (int col = 0; col < width; ++col)
    {
      float sum = 0.0f;
      for (int dy = -radius; dy <= radius; ++dy)
      {
        for (int dx = -radius; dx <= radius; ++dx)
        {
          const int y = row + dy;
          const int x = col + dx;
          if (x >= 0 && x < width && y >= 0 && y < height)
          {
            const int inIdx = y * width + x;
            const int fIdx = (dy + radius) * filterSize + (dx + radius);
            sum += input[inIdx] * filter[fIdx];
          }
        }
      }
      output[row * width + col] = sum;
    }
  }
}

// 2D Basic Kernel
__global__ void kernel2D_basic(float *input, float *output, float *filter,
                               int width, int height, int filterSize)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= width || row >= height)
    return;

  float sum = 0.0f;
  const int radius = filterSize >> 1;

  for (int dy = -radius; dy <= radius; ++dy)
  {
    for (int dx = -radius; dx <= radius; ++dx)
    {
      const int y = row + dy;
      const int x = col + dx;
      if (x >= 0 && x < width && y >= 0 && y < height)
      {
        const int inIdx = y * width + x;
        const int fIdx = (dy + radius) * filterSize + (dx + radius);
        sum += input[inIdx] * filter[fIdx];
      }
    }
  }

  output[row * width + col] = sum;
}

__constant__ float d_filter2D_const[filterSize2D * filterSize2D];
// 2D Optimized Kernel
/**
 * Loop unrolling to the 2D convolution kernel.
 **/
__global__ void kernel2D_unrolling(float *input, float *output, float *filter,
                                   int width, int height, int filterSize)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= width || row >= height)
    return;

  float sum = 0.0f;
  const int radius = filterSize >> 1;

#pragma unroll
  for (int dy = -radius; dy <= radius; ++dy)
  {
#pragma unroll
    for (int dx = -radius; dx <= radius; ++dx)
    {
      const int y = row + dy;
      const int x = col + dx;
      if (x >= 0 && x < width && y >= 0 && y < height)
      {
        const int inIdx = y * width + x;
        const int fIdx = (dy + radius) * filterSize + (dx + radius);
        sum += input[inIdx] * filter[fIdx];
      }
    }
  }

  output[row * width + col] = sum;
}

// 2D Optimized Kernel
/**
 * Loop unrolling to the 2D convolution kernel.
 * Constant memory for filter.
 **/
__global__ void kernel2D_constant(float *input, float *output, int width, int height, int filterSize)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= width || row >= height)
    return;

  float sum = 0.0f;
  const int radius = filterSize >> 1;

#pragma unroll
  for (int dy = -radius; dy <= radius; ++dy)
  {
#pragma unroll
    for (int dx = -radius; dx <= radius; ++dx)
    {
      const int y = row + dy;
      const int x = col + dx;
      if (x >= 0 && x < width && y >= 0 && y < height)
      {
        const int inIdx = y * width + x;
        const int fIdx = (dy + radius) * filterSize + (dx + radius);
        sum += input[inIdx] * d_filter2D_const[fIdx];
      }
    }
  }

  output[row * width + col] = sum;
}

// 2D Optimized Kernel
/**
 * Loop unrolling to the 2D convolution kernel.
 * Constant memory for filter.
 * Tiling optimization.
 **/
__global__ void kernel2D_opt(float *input, float *output, int width, int height, int filterSize)
{
  const int radius = filterSize >> 1;
  __shared__ float tile[TILE_SIZE + 2 * filterRadius2D][TILE_SIZE + 2 * filterRadius2D];

  const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  const int row = blockIdx.y * TILE_SIZE + threadIdx.y;

  int tx = threadIdx.x + radius;
  int ty = threadIdx.y + radius;

  // Load data into shared memory
  if (col < width && row < height)
  {
    tile[ty][tx] = input[row * width + col];
  }
  else
  {
    tile[ty][tx] = 0.0f;
  }
  // Load halo regions
  // Left halo
  // Load left halo
  if (threadIdx.x < filterRadius2D)
  {
    int haloCol = col - filterRadius2D;
    if (row < width && haloCol >= 0)
    {
      tile[ty][threadIdx.x] = input[row * width + haloCol];
    }
    else
    {
      tile[ty][threadIdx.x] = 0.0f;
    }
  }

  // Load right halo
  if (threadIdx.x < filterRadius2D)
  {
    int haloCol = col + TILE_SIZE;
    if (row < width && haloCol < width)
    {
      tile[ty][tx + TILE_SIZE] = input[row * width + haloCol];
    }
    else
    {
      tile[ty][tx + TILE_SIZE] = 0.0f;
    }
  }

  // Load top halo
  if (threadIdx.y < filterRadius2D)
  {
    int haloRow = row - filterRadius2D;
    if (haloRow >= 0 && col < width)
    {
      tile[threadIdx.y][tx] = input[haloRow * width + col];
    }
    else
    {
      tile[threadIdx.y][tx] = 0.0f;
    }
  }

  // Load bottom halo
  if (threadIdx.y < filterRadius2D)
  {
    int haloRow = row + TILE_SIZE;
    if (haloRow < width && col < width)
    {
      tile[ty + TILE_SIZE][tx] = input[haloRow * width + col];
    }
    else
    {
      tile[ty + TILE_SIZE][tx] = 0.0f;
    }
  }

  // Load corner halos
  // Top-left
  if (threadIdx.x < filterRadius2D && threadIdx.y < filterRadius2D)
  {
    int haloRow = row - filterRadius2D;
    int haloCol = col - filterRadius2D;
    if (haloRow >= 0 && haloCol >= 0)
    {
      tile[threadIdx.y][threadIdx.x] = input[haloRow * width + haloCol];
    }
    else
    {
      tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
  }

  // Top-right
  if (threadIdx.x < filterRadius2D && threadIdx.y < filterRadius2D)
  {
    int haloRow = row - filterRadius2D;
    int haloCol = col + TILE_SIZE;
    if (haloRow >= 0 && haloCol < width)
    {
      tile[threadIdx.y][tx + TILE_SIZE] = input[haloRow * width + haloCol];
    }
    else
    {
      tile[threadIdx.y][tx + TILE_SIZE] = 0.0f;
    }
  }

  // Bottom-left
  if (threadIdx.x < filterRadius2D && threadIdx.y < filterRadius2D)
  {
    int haloRow = row + TILE_SIZE;
    int haloCol = col - filterRadius2D;
    if (haloRow < width && haloCol >= 0)
    {
      tile[ty + TILE_SIZE][threadIdx.x] = input[haloRow * width + haloCol];
    }
    else
    {
      tile[ty + TILE_SIZE][threadIdx.x] = 0.0f;
    }
  }

  // Bottom-right
  if (threadIdx.x < filterRadius2D && threadIdx.y < filterRadius2D)
  {
    int haloRow = row + TILE_SIZE;
    int haloCol = col + TILE_SIZE;
    if (haloRow < width && haloCol < width)
    {
      tile[ty + TILE_SIZE][tx + TILE_SIZE] = input[haloRow * width + haloCol];
    }
    else
    {
      tile[ty + TILE_SIZE][tx + TILE_SIZE] = 0.0f;
    }
  }
  __syncthreads();

  // Perform convolution
  float sum = 0.0f;
#pragma unroll
  for (int dy = -radius; dy <= radius; ++dy)
  {
#pragma unroll
    for (int dx = -radius; dx <= radius; ++dx)
    {
      const int fIdx = (dy + radius) * filterSize + (dx + radius);
      sum += tile[ty + dy][tx + dx] * d_filter2D_const[fIdx];
    }
  }
  output[row * width + col] = sum;
}

// 3D Convolution - CPU version
__host__ void kernel3D_cpu(const float *input, float *output,
                           const float *filter, int width, int height,
                           int depth, int filterSize)
{
  const int radius = filterSize >> 1;
  for (int z = 0; z < depth; ++z)
  {
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        float sum = 0.0f;
        for (int dz = -radius; dz <= radius; ++dz)
        {
          const int nz = z + dz;
          if (nz < 0 || nz >= depth)
            continue;
          for (int dy = -radius; dy <= radius; ++dy)
          {
            const int ny = y + dy;
            if (ny < 0 || ny >= height)
              continue;
            for (int dx = -radius; dx <= radius; ++dx)
            {
              const int nx = x + dx;
              if (nx < 0 || nx >= width)
                continue;
              const int inIdx = nz * width * height + ny * width + nx;
              const int fIdx =
                  (dz + radius) * filterSize * filterSize +
                  (dy + radius) * filterSize + (dx + radius);
              sum += input[inIdx] * filter[fIdx];
            }
          }
        }
        output[z * width * height + y * width + x] = sum;
      }
    }
  }
}

// 3D Basic Kernel
__global__ void kernel3D_basic(float *input, float *output, float *filter,
                               int width, int height, int depth,
                               int filterSize)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= width || y >= height || z >= depth)
    return;

  float sum = 0.0f;
  const int radius = filterSize >> 1;

  for (int dz = -radius; dz <= radius; ++dz)
  {
    const int nz = z + dz;
    if (nz < 0 || nz >= depth)
      continue;
    for (int dy = -radius; dy <= radius; ++dy)
    {
      const int ny = y + dy;
      if (ny < 0 || ny >= height)
        continue;
      for (int dx = -radius; dx <= radius; ++dx)
      {
        const int nx = x + dx;
        if (nx < 0 || nx >= width)
          continue;
        const int inIdx = nz * width * height + ny * width + nx;
        const int fIdx =
            (dz + radius) * filterSize * filterSize +
            (dy + radius) * filterSize + (dx + radius);
        sum += input[inIdx] * filter[fIdx];
      }
    }
  }

  output[z * width * height + y * width + x] = sum;
}

// 3D Optimized Kernel
/**
 * Loop unrolling to the 3D convolution kernel.
 **/
__global__ void kernel3D_unrolling(float *input, float *output, float *filter,
                                   int width, int height, int depth,
                                   int filterSize)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= width || y >= height || z >= depth)
    return;

  float sum = 0.0f;
  const int radius = filterSize >> 1;
#pragma unroll
  for (int dz = -radius; dz <= radius; ++dz)
  {
    const int nz = z + dz;
    if (nz < 0 || nz >= depth)
      continue;
#pragma unroll
    for (int dy = -radius; dy <= radius; ++dy)
    {
      const int ny = y + dy;
      if (ny < 0 || ny >= height)
        continue;

#pragma unroll
      for (int dx = -radius; dx <= radius; ++dx)
      {
        const int nx = x + dx;
        if (nx < 0 || nx >= width)
          continue;
        const int inIdx = nz * width * height + ny * width + nx;
        const int fIdx =
            (dz + radius) * filterSize * filterSize +
            (dy + radius) * filterSize + (dx + radius);
        sum += input[inIdx] * filter[fIdx];
      }
    }
  }

  output[z * width * height + y * width + x] = sum;
}

__constant__ float d_filter3D_const[filterSize3D * filterSize3D * filterSize3D];
// 3D Optimized Kernel
/**
 * Loop unrolling to the 3D convolution kernel.
 * Constant memory for filter.
 * Tiling optimization.
 **/
__global__ void kernel3D_opt(float *input, float *output, float *filter,
                             int width, int height, int depth,
                             int filterSize)
{
  const int radius = filterSize >> 1;
  __shared__ float tile[TILE_SIZE + 2 * filterRadius3D][TILE_SIZE + 2 * filterRadius3D][TILE_SIZE + 2 * filterRadius3D];

  // Global coordinates
  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;
  int z = blockIdx.z * TILE_SIZE + threadIdx.z;

  // Shared memory coordinates (with halo offset)
  int tx = threadIdx.x + radius;
  int ty = threadIdx.y + radius;
  int tz = threadIdx.z + radius;

  // Load center data
  if (x < width && y < width && z < width)
  {
    tile[tz][ty][tx] = input[z * width * width + y * width + x];
  }
  else
  {
    tile[tz][ty][tx] = 0.0f;
  }

  // Load face halos (6 faces)
  // -X face
  if (threadIdx.x < radius)
  {
    int haloX = x - radius;
    if (haloX >= 0 && y < width && z < width)
    {
      tile[tz][ty][threadIdx.x] = input[z * width * width + y * width + haloX];
    }
    else
    {
      tile[tz][ty][threadIdx.x] = 0.0f;
    }
  }

  // +X face
  if (threadIdx.x < radius)
  {
    int haloX = x + TILE_SIZE;
    if (haloX < width && y < width && z < width)
    {
      tile[tz][ty][tx + TILE_SIZE] = input[z * width * width + y * width + haloX];
    }
    else
    {
      tile[tz][ty][tx + TILE_SIZE] = 0.0f;
    }
  }

  // -Y face
  if (threadIdx.y < radius)
  {
    int haloY = y - radius;
    if (x < width && haloY >= 0 && z < width)
    {
      tile[tz][threadIdx.y][tx] = input[z * width * width + haloY * width + x];
    }
    else
    {
      tile[tz][threadIdx.y][tx] = 0.0f;
    }
  }

  // +Y face
  if (threadIdx.y < radius)
  {
    int haloY = y + TILE_SIZE;
    if (x < width && haloY < width && z < width)
    {
      tile[tz][ty + TILE_SIZE][tx] = input[z * width * width + haloY * width + x];
    }
    else
    {
      tile[tz][ty + TILE_SIZE][tx] = 0.0f;
    }
  }

  // -Z face
  if (threadIdx.z < radius)
  {
    int haloZ = z - radius;
    if (x < width && y < width && haloZ >= 0)
    {
      tile[threadIdx.z][ty][tx] = input[haloZ * width * width + y * width + x];
    }
    else
    {
      tile[threadIdx.z][ty][tx] = 0.0f;
    }
  }

  // +Z face
  if (threadIdx.z < radius)
  {
    int haloZ = z + TILE_SIZE;
    if (x < width && y < width && haloZ < width)
    {
      tile[tz + TILE_SIZE][ty][tx] = input[haloZ * width * width + y * width + x];
    }
    else
    {
      tile[tz + TILE_SIZE][ty][tx] = 0.0f;
    }
  }

  // Load edge halos (12 edges) - simplified approach
  // We'll let threads with both conditions load edges

  // X-Y edges
  if (threadIdx.x < radius && threadIdx.y < radius)
  {
    // -X, -Y
    int haloX = x - radius;
    int haloY = y - radius;
    if (haloX >= 0 && haloY >= 0 && z < width)
    {
      tile[tz][threadIdx.y][threadIdx.x] = input[z * width * width + haloY * width + haloX];
    }
    else
    {
      tile[tz][threadIdx.y][threadIdx.x] = 0.0f;
    }

    // +X, -Y
    haloX = x + TILE_SIZE;
    haloY = y - radius;
    if (haloX < width && haloY >= 0 && z < width)
    {
      tile[tz][threadIdx.y][tx + TILE_SIZE] = input[z * width * width + haloY * width + haloX];
    }
    else
    {
      tile[tz][threadIdx.y][tx + TILE_SIZE] = 0.0f;
    }

    // -X, +Y
    haloX = x - radius;
    haloY = y + TILE_SIZE;
    if (haloX >= 0 && haloY < width && z < width)
    {
      tile[tz][ty + TILE_SIZE][threadIdx.x] = input[z * width * width + haloY * width + haloX];
    }
    else
    {
      tile[tz][ty + TILE_SIZE][threadIdx.x] = 0.0f;
    }

    // +X, +Y
    haloX = x + TILE_SIZE;
    haloY = y + TILE_SIZE;
    if (haloX < width && haloY < width && z < width)
    {
      tile[tz][ty + TILE_SIZE][tx + TILE_SIZE] = input[z * width * width + haloY * width + haloX];
    }
    else
    {
      tile[tz][ty + TILE_SIZE][tx + TILE_SIZE] = 0.0f;
    }
  }

  // Y-Z edges
  if (threadIdx.y < radius && threadIdx.z < radius)
  {
    // -Y, -Z
    int haloY = y - radius;
    int haloZ = z - radius;
    if (x < width && haloY >= 0 && haloZ >= 0)
    {
      tile[threadIdx.z][threadIdx.y][tx] = input[haloZ * width * width + haloY * width + x];
    }
    else
    {
      tile[threadIdx.z][threadIdx.y][tx] = 0.0f;
    }

    // +Y, -Z
    haloY = y + TILE_SIZE;
    haloZ = z - radius;
    if (x < width && haloY < width && haloZ >= 0)
    {
      tile[threadIdx.z][ty + TILE_SIZE][tx] = input[haloZ * width * width + haloY * width + x];
    }
    else
    {
      tile[threadIdx.z][ty + TILE_SIZE][tx] = 0.0f;
    }

    // -Y, +Z
    haloY = y - radius;
    haloZ = z + TILE_SIZE;
    if (x < width && haloY >= 0 && haloZ < width)
    {
      tile[tz + TILE_SIZE][threadIdx.y][tx] = input[haloZ * width * width + haloY * width + x];
    }
    else
    {
      tile[tz + TILE_SIZE][threadIdx.y][tx] = 0.0f;
    }

    // +Y, +Z
    haloY = y + TILE_SIZE;
    haloZ = z + TILE_SIZE;
    if (x < width && haloY < width && haloZ < width)
    {
      tile[tz + TILE_SIZE][ty + TILE_SIZE][tx] = input[haloZ * width * width + haloY * width + x];
    }
    else
    {
      tile[tz + TILE_SIZE][ty + TILE_SIZE][tx] = 0.0f;
    }
  }

  // X-Z edges
  if (threadIdx.x < radius && threadIdx.z < radius)
  {
    // -X, -Z
    int haloX = x - radius;
    int haloZ = z - radius;
    if (haloX >= 0 && y < width && haloZ >= 0)
    {
      tile[threadIdx.z][ty][threadIdx.x] = input[haloZ * width * width + y * width + haloX];
    }
    else
    {
      tile[threadIdx.z][ty][threadIdx.x] = 0.0f;
    }

    // +X, -Z
    haloX = x + TILE_SIZE;
    haloZ = z - radius;
    if (haloX < width && y < width && haloZ >= 0)
    {
      tile[threadIdx.z][ty][tx + TILE_SIZE] = input[haloZ * width * width + y * width + haloX];
    }
    else
    {
      tile[threadIdx.z][ty][tx + TILE_SIZE] = 0.0f;
    }

    // -X, +Z
    haloX = x - radius;
    haloZ = z + TILE_SIZE;
    if (haloX >= 0 && y < width && haloZ < width)
    {
      tile[tz + TILE_SIZE][ty][threadIdx.x] = input[haloZ * width * width + y * width + haloX];
    }
    else
    {
      tile[tz + TILE_SIZE][ty][threadIdx.x] = 0.0f;
    }

    // +X, +Z
    haloX = x + TILE_SIZE;
    haloZ = z + TILE_SIZE;
    if (haloX < width && y < width && haloZ < width)
    {
      tile[tz + TILE_SIZE][ty][tx + TILE_SIZE] = input[haloZ * width * width + y * width + haloX];
    }
    else
    {
      tile[tz + TILE_SIZE][ty][tx + TILE_SIZE] = 0.0f;
    }
  }

  // Load corner halos (8 corners)
  if (threadIdx.x < radius && threadIdx.y < radius && threadIdx.z < radius)
  {
    // All 8 corners
    int coords[8][3] = {
        {x - radius, y - radius, z - radius}, // ---
        {x + TILE_SIZE, y - radius, z - radius},     // +--
        {x - radius, y + TILE_SIZE, z - radius},     // -+-
        {x + TILE_SIZE, y + TILE_SIZE, z - radius},         // ++-
        {x - radius, y - radius, z + TILE_SIZE},     // --+
        {x + TILE_SIZE, y - radius, z + TILE_SIZE},         // +-+
        {x - radius, y + TILE_SIZE, z + TILE_SIZE},         // -++
        {x + TILE_SIZE, y + TILE_SIZE, z + TILE_SIZE}              // +++
    };

    int tileCoords[8][3] = {
        {threadIdx.x, threadIdx.y, threadIdx.z},
        {tx + TILE_SIZE, threadIdx.y, threadIdx.z},
        {threadIdx.x, ty + TILE_SIZE, threadIdx.z},
        {tx + TILE_SIZE, ty + TILE_SIZE, threadIdx.z},
        {threadIdx.x, threadIdx.y, tz + TILE_SIZE},
        {tx + TILE_SIZE, threadIdx.y, tz + TILE_SIZE},
        {threadIdx.x, ty + TILE_SIZE, tz + TILE_SIZE},
        {tx + TILE_SIZE, ty + TILE_SIZE, tz + TILE_SIZE}};

    for (int i = 0; i < 8; i++)
    {
      int haloX = coords[i][0];
      int haloY = coords[i][1];
      int haloZ = coords[i][2];

      if (haloX >= 0 && haloX < width && haloY >= 0 && haloY < width && haloZ >= 0 && haloZ < width)
      {
        tile[tileCoords[i][2]][tileCoords[i][1]][tileCoords[i][0]] =
            input[haloZ * width * width + haloY * width + haloX];
      }
      else
      {
        tile[tileCoords[i][2]][tileCoords[i][1]][tileCoords[i][0]] = 0.0f;
      }
    }
  }

  __syncthreads();

  // Compute convolution
  if (x < width && y < width && z < width)
  {
    float sum = 0.0f;

    int idx = 0;
#pragma unroll
    for (int kz = 0; kz < 3; kz++)
    {
#pragma unroll
      for (int ky = 0; ky < 3; ky++)
      {
#pragma unroll
        for (int kx = 0; kx < 3; kx++)
        {
          sum += tile[tz + kz - 1][ty + ky - 1][tx + kx - 1] * d_filter3D_const[idx++];
        }
      }
    }

    output[z * width * height + y * width + x] = sum;
  }
}

__host__ void check_result2D(const float *ref, const float *test, int size)
{
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < size; i++)
  {
    double this_diff = std::fabs(ref[i] - test[i]);
    if (this_diff > THRESHOLD)
    {
      numdiffs++;
      if (this_diff > maxdiff)
      {
        maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0)
  {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  }
  else
  {
    cout << "No differences found between base and test versions\n";
  }
}

__host__ void check_result3D(const float *ref, const float *test, int size)
{
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < size; i++)
  {
    double this_diff = std::fabs(ref[i] - test[i]);
    if (this_diff > THRESHOLD * 100)
    { // Relaxed threshold for 3D
      numdiffs++;
      if (this_diff > maxdiff)
      {
        maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0)
  {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD * 100
         << "; Max Diff = " << maxdiff << endl;
  }
  else
  {
    cout << "No differences found between base and test versions\n";
  }
}

double rtclock()
{ // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
  {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main()
{
  const int elems2D = static_cast<int>(N * N);
  const int elems3D = static_cast<int>(N * N * N);
  const int filterElems2D = filterSize2D * filterSize2D;
  const int filterElems3D = filterSize3D * filterSize3D * filterSize3D;

  float *h_in2D = (float *)malloc(elems2D * sizeof(float));
  float *h_out2D = (float *)malloc(elems2D * sizeof(float));
  float *h_out2D_cpu = (float *)malloc(elems2D * sizeof(float));
  float *h_filter2D = (float *)malloc(filterElems2D * sizeof(float));

  float *h_in3D = (float *)malloc(elems3D * sizeof(float));
  float *h_out3D = (float *)malloc(elems3D * sizeof(float));
  float *h_out3D_cpu = (float *)malloc(elems3D * sizeof(float));
  float *h_filter3D = (float *)malloc(filterElems3D * sizeof(float));

  for (int i = 0; i < elems2D; ++i)
  {
    h_in2D[i] = static_cast<float>((i % 13) / 13.0f);
  }
  for (int i = 0; i < filterElems2D; ++i)
  {
    h_filter2D[i] = 1.0f / filterElems2D;
  }

  for (int i = 0; i < elems3D; ++i)
  {
    h_in3D[i] = static_cast<float>((i % 17) / 17.0f);
  }
  for (int i = 0; i < filterElems3D; ++i)
  {
    h_filter3D[i] = 1.0f / filterElems3D;
  }

  float *d_in2D, *d_out2D, *d_filter2D;
  float *d_in3D, *d_out3D, *d_filter3D;
  cudaCheckError(cudaMalloc(&d_in2D, elems2D * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_out2D, elems2D * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_filter2D, filterElems2D * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_in3D, elems3D * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_out3D, elems3D * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_filter3D, filterElems3D * sizeof(float)));

  cudaCheckError(cudaMemcpy(d_in2D, h_in2D, elems2D * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_filter2D, h_filter2D,
                            filterElems2D * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpyToSymbol(d_filter2D_const, h_filter2D, filterElems2D * sizeof(float)));

  cudaCheckError(cudaMemcpy(d_in3D, h_in3D, elems3D * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_filter3D, h_filter3D,
                            filterElems3D * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpyToSymbol(d_filter3D_const, h_filter3D, filterElems3D * sizeof(float)));

  cudaEvent_t start, stop;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));
  float kernel_time = 0.0f;

  dim3 block2D(16, 16);
  dim3 grid2D((N + block2D.x - 1) / block2D.x,
              (N + block2D.y - 1) / block2D.y);

  dim3 block2D_opt(TILE_SIZE, TILE_SIZE);
  dim3 grid2D_opt((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

  cout << "            Kernel 2D Benchmarking:\n";
  cout << "--------------------------------------------------\n";
  // CPU Reference
  double cpu_start = rtclock();
  kernel2D_cpu(h_in2D, h_out2D_cpu, h_filter2D, N, N, filterSize2D);
  double cpu_end = rtclock();
  cout << "Kernel_2D CPU Time (ms): " << (cpu_end - cpu_start) * 1000 << "\n";

  // Launch Basic 2D Kernel
  cudaCheckError(cudaEventRecord(start));
  kernel2D_basic<<<grid2D, block2D>>>(d_in2D, d_out2D, d_filter2D, N, N,
                                      filterSize2D);
  cudaCheckError(cudaEventRecord(stop));
  cudaCheckError(cudaEventSynchronize(stop));
  cudaCheckError(cudaEventElapsedTime(&kernel_time, start, stop));
  cudaCheckError(cudaMemcpy(h_out2D, d_out2D, elems2D * sizeof(float),
                            cudaMemcpyDeviceToHost));
  cout << "Kernel2D Basic GPU time (ms): " << kernel_time << "\n";
  check_result2D(h_out2D_cpu, h_out2D, elems2D);
  // Launch Optimized 2D Kernel
  cudaCheckError(cudaEventRecord(start));
  kernel2D_unrolling<<<grid2D_opt, block2D_opt>>>(d_in2D, d_out2D, d_filter2D, N, N,
                                          filterSize2D);
  cudaCheckError(cudaEventRecord(stop));
  cudaCheckError(cudaEventSynchronize(stop));
  cudaCheckError(cudaEventElapsedTime(&kernel_time, start, stop));
  cudaCheckError(cudaMemcpy(h_out2D, d_out2D, elems2D * sizeof(float),
                            cudaMemcpyDeviceToHost));
  cout << "Kernel2D [U] time (ms): " << kernel_time << "\n";
  check_result2D(h_out2D_cpu, h_out2D, elems2D);
  cudaCheckError(cudaEventRecord(start));
  kernel2D_opt<<<grid2D_opt, block2D_opt>>>(d_in2D, d_out2D, N, N,
                                            filterSize2D);
  cudaCheckError(cudaEventRecord(stop));
  cudaCheckError(cudaEventSynchronize(stop));
  cudaCheckError(cudaEventElapsedTime(&kernel_time, start, stop));
  cudaCheckError(cudaMemcpy(h_out2D, d_out2D, elems2D * sizeof(float),
                            cudaMemcpyDeviceToHost));
  cout << "Kernel2D [U + CF + T] time (ms): " << kernel_time << "\n";
  check_result2D(h_out2D_cpu, h_out2D, elems2D);

  // Launch Basic 3D Kernel
  dim3 block3D(4, 4, 4);
  dim3 grid3D((N + block3D.x - 1) / block3D.x,
              (N + block3D.y - 1) / block3D.y,
              (N + block3D.z - 1) / block3D.z);
  dim3 block3D_opt(TILE_SIZE, TILE_SIZE, TILE_SIZE);
  dim3 grid3D_opt((N + TILE_SIZE - 1) / TILE_SIZE,
                  (N + TILE_SIZE - 1) / TILE_SIZE,
                  (N + TILE_SIZE - 1) / TILE_SIZE);

  cout << "            Kernel 3D Benchmarking:\n";
  cout << "--------------------------------------------------\n";
  // CPU Reference
  cpu_start = rtclock();
  kernel3D_cpu(h_in3D, h_out3D_cpu, h_filter3D, N, N, N, filterSize3D);
  cpu_end = rtclock();
  cout << "Kernel_3D CPU Time (ms): " << (cpu_end - cpu_start) * 1000 << "\n";

  cudaCheckError(cudaEventRecord(start));
  kernel3D_basic<<<grid3D, block3D>>>(d_in3D, d_out3D, d_filter3D, N, N, N,
                                      filterSize3D);
  cudaCheckError(cudaEventRecord(stop));
  cudaCheckError(cudaEventSynchronize(stop));
  cudaCheckError(cudaEventElapsedTime(&kernel_time, start, stop));
  cudaCheckError(cudaMemcpy(h_out3D, d_out3D, elems3D * sizeof(float),
                            cudaMemcpyDeviceToHost));
  cout << "Kernel3D Basic GPU time (ms): " << kernel_time << "\n";
  check_result3D(h_out3D_cpu, h_out3D, elems3D);

  cudaCheckError(cudaEventRecord(start));
  kernel3D_unrolling<<<grid3D_opt, block3D_opt>>>(d_in3D, d_out3D, d_filter3D, N, N, N,
                                          filterSize3D);
  cudaCheckError(cudaEventRecord(stop));
  cudaCheckError(cudaEventSynchronize(stop));
  cudaCheckError(cudaEventElapsedTime(&kernel_time, start, stop));
  cudaCheckError(cudaMemcpy(h_out3D, d_out3D, elems3D * sizeof(float),
                            cudaMemcpyDeviceToHost));
  cout << "Kernel3D [U] time (ms): " << kernel_time << "\n";
  check_result3D(h_out3D_cpu, h_out3D, elems3D);

  cudaCheckError(cudaEventRecord(start));
  kernel3D_opt<<<grid3D_opt, block3D_opt>>>(d_in3D, d_out3D, d_filter3D, N, N, N,
                                            filterSize3D);
  cudaCheckError(cudaEventRecord(stop));
  cudaCheckError(cudaEventSynchronize(stop));
  cudaCheckError(cudaEventElapsedTime(&kernel_time, start, stop));
  cudaCheckError(cudaMemcpy(h_out3D, d_out3D, elems3D * sizeof(float),
                            cudaMemcpyDeviceToHost));
  cout << "Kernel3D [U + CF + T] time (ms): " << kernel_time << "\n";
  check_result3D(h_out3D_cpu, h_out3D, elems3D);

  cudaCheckError(cudaFree(d_in2D));
  cudaCheckError(cudaFree(d_out2D));
  cudaCheckError(cudaFree(d_filter2D));
  cudaCheckError(cudaFree(d_in3D));
  cudaCheckError(cudaFree(d_out3D));
  cudaCheckError(cudaFree(d_filter3D));
  cudaCheckError(cudaEventDestroy(start));
  cudaCheckError(cudaEventDestroy(stop));

  free(h_in2D);
  free(h_out2D);
  free(h_filter2D);
  free(h_in3D);
  free(h_out3D);
  free(h_filter3D);

  return EXIT_SUCCESS;
}
