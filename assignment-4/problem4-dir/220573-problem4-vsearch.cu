    #include <cstdlib>
    #include <cuda.h>
    #include <iostream>
    #include <numeric>
    #include <sys/time.h>
    #include <vector>
    #include <algorithm>

    #define THRESHOLD (0.001f)  // More reasonable threshold
    using std::cerr;
    using std::cout;
    using std::endl;

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

    const uint64_t N = (1 << 6);  // 64
    const int FILTER_SIZE = 3;
    const int FILTER_RADIUS = FILTER_SIZE / 2;

    const int TILE_SIZES_2D[] = {2, 4, 8, 16};
    const int NUM_TILE_SIZES_2D = 4;

    const int TILE_SIZES_3D[] = {2, 4, 8, 16};
    const int NUM_TILE_SIZES_3D = 4;

    const int MAX_TILE_2D = 32;
    const int MAX_TILE_3D = 20;

    // 2D Basic kernel
    __global__ void kernel2D_basic(const float* input, const float* filter, 
                                float* output, int width, int filterSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        int filterRadius = filterSize / 2;
        
        for (int fRow = 0; fRow < filterSize; fRow++) {
        for (int fCol = 0; fCol < filterSize; fCol++) {
            int inRow = row + fRow - filterRadius;
            int inCol = col + fCol - filterRadius;
            
            float inputValue = 0.0f;
            if (inRow >= 0 && inRow < width && inCol >= 0 && inCol < width) {
            inputValue = input[inRow * width + inCol];
            }
            
            float filterValue = filter[fRow * filterSize + fCol];
            sum += inputValue * filterValue;
        }
        }
        
        output[row * width + col] = sum;
    }
    }

    // 2D Optimized kernel - FIXED VERSION
    __constant__ float d_filter2D[9];

template<int TILE_SIZE>
__global__ void kernel2D_opt_templated(const float* input, float* output, 
                                    int width, int filterSize) {
    const int radius = filterSize >> 1;
    __shared__ float tile[TILE_SIZE + 2 * FILTER_RADIUS][TILE_SIZE + 2 * FILTER_RADIUS];

    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    int tx = threadIdx.x + radius;
    int ty = threadIdx.y + radius;

    // Load core tile
    if (col < width && row < width) {
        tile[ty][tx] = input[row * width + col];
    } else {
        tile[ty][tx] = 0.0f;
    }

    // Halos
    if (threadIdx.x < radius) {
        int haloCol = col - radius;
        tile[ty][threadIdx.x] = (row < width && haloCol >= 0) ? input[row * width + haloCol] : 0.0f;
        haloCol = col + TILE_SIZE;
        tile[ty][tx + TILE_SIZE] = (row < width && haloCol < width) ? input[row * width + haloCol] : 0.0f;
    }

    if (threadIdx.y < radius) {
        int haloRow = row - radius;
        tile[threadIdx.y][tx] = (haloRow >= 0 && col < width) ? input[haloRow * width + col] : 0.0f;
        haloRow = row + TILE_SIZE;
        tile[ty + TILE_SIZE][tx] = (haloRow < width && col < width) ? input[haloRow * width + col] : 0.0f;
    }

    if (threadIdx.x < radius && threadIdx.y < radius) {
        int haloRow = row - radius;
        int haloCol = col - radius;
        tile[threadIdx.y][threadIdx.x] = (haloRow >= 0 && haloCol >= 0) ? input[haloRow * width + haloCol] : 0.0f;

        haloRow = row - radius;
        haloCol = col + TILE_SIZE;
        tile[threadIdx.y][tx + TILE_SIZE] = (haloRow >= 0 && haloCol < width) ? input[haloRow * width + haloCol] : 0.0f;

        haloRow = row + TILE_SIZE;
        haloCol = col - radius;
        tile[ty + TILE_SIZE][threadIdx.x] = (haloRow < width && haloCol >= 0) ? input[haloRow * width + haloCol] : 0.0f;

        haloRow = row + TILE_SIZE;
        haloCol = col + TILE_SIZE;
        tile[ty + TILE_SIZE][tx + TILE_SIZE] = (haloRow < width && haloCol < width) ? input[haloRow * width + haloCol] : 0.0f;
    }

    __syncthreads();

    if (row < width && col < width) {
        float sum = 0.0f;
#pragma unroll
        for (int dy = -radius; dy <= radius; ++dy) {
#pragma unroll
            for (int dx = -radius; dx <= radius; ++dx) {
                const int fIdx = (dy + radius) * filterSize + (dx + radius);
                sum += tile[ty + dy][tx + dx] * d_filter2D[fIdx];
            }
        }
        output[row * width + col] = sum;
    }
}

    // 3D Basic kernel
    __global__ void kernel3D_basic(const float* input, const float* filter, 
                                float* output, int width, int filterSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < width && z < width) {
        float sum = 0.0f;
        int filterRadius = filterSize / 2;
        
        for (int fz = 0; fz < filterSize; fz++) {
        for (int fy = 0; fy < filterSize; fy++) {
            for (int fx = 0; fx < filterSize; fx++) {
            int inX = x + fx - filterRadius;
            int inY = y + fy - filterRadius;
            int inZ = z + fz - filterRadius;
            
            float inputValue = 0.0f;
            if (inX >= 0 && inX < width && inY >= 0 && inY < width && 
                inZ >= 0 && inZ < width) {
                inputValue = input[inZ * width * width + inY * width + inX];
            }
            
            float filterValue = filter[fz * filterSize * filterSize + 
                                        fy * filterSize + fx];
            sum += inputValue * filterValue;
            }
        }
        }
        
        output[z * width * width + y * width + x] = sum;
    }
    }

    // 3D Optimized kernel - FIXED VERSION
    __constant__ float d_filter3D[27];

template<int TILE_SIZE>
__global__ void kernel3D_opt_templated(const float* input, float* output, 
                                    int width, int filterSize) {
    const int radius = filterSize >> 1;
    const int SHARED_SIZE = TILE_SIZE + 2 * FILTER_RADIUS;
    __shared__ float tile[SHARED_SIZE][SHARED_SIZE][SHARED_SIZE];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int z = blockIdx.z * TILE_SIZE + threadIdx.z;

    int tx = threadIdx.x + radius;
    int ty = threadIdx.y + radius;
    int tz = threadIdx.z + radius;

    tile[tz][ty][tx] = (x < width && y < width && z < width)
        ? input[z * width * width + y * width + x] : 0.0f;

    if (threadIdx.x < radius) {
        int haloX = x - radius;
        tile[tz][ty][threadIdx.x] = (haloX >= 0 && y < width && z < width) ? input[z * width * width + y * width + haloX] : 0.0f;
        haloX = x + TILE_SIZE;
        tile[tz][ty][tx + TILE_SIZE] = (haloX < width && y < width && z < width) ? input[z * width * width + y * width + haloX] : 0.0f;
    }

    if (threadIdx.y < radius) {
        int haloY = y - radius;
        tile[tz][threadIdx.y][tx] = (x < width && haloY >= 0 && z < width) ? input[z * width * width + haloY * width + x] : 0.0f;
        haloY = y + TILE_SIZE;
        tile[tz][ty + TILE_SIZE][tx] = (x < width && haloY < width && z < width) ? input[z * width * width + haloY * width + x] : 0.0f;
    }

    if (threadIdx.z < radius) {
        int haloZ = z - radius;
        tile[threadIdx.z][ty][tx] = (x < width && y < width && haloZ >= 0) ? input[haloZ * width * width + y * width + x] : 0.0f;
        haloZ = z + TILE_SIZE;
        tile[tz + TILE_SIZE][ty][tx] = (x < width && y < width && haloZ < width) ? input[haloZ * width * width + y * width + x] : 0.0f;
    }

    // Edges
    if (threadIdx.x < radius && threadIdx.y < radius) {
        int haloX = x - radius;
        int haloY = y - radius;
        tile[tz][threadIdx.y][threadIdx.x] = (haloX >= 0 && haloY >= 0 && z < width) ? input[z * width * width + haloY * width + haloX] : 0.0f;

        haloX = x + TILE_SIZE; haloY = y - radius;
        tile[tz][threadIdx.y][tx + TILE_SIZE] = (haloX < width && haloY >= 0 && z < width) ? input[z * width * width + haloY * width + haloX] : 0.0f;

        haloX = x - radius; haloY = y + TILE_SIZE;
        tile[tz][ty + TILE_SIZE][threadIdx.x] = (haloX >= 0 && haloY < width && z < width) ? input[z * width * width + haloY * width + haloX] : 0.0f;

        haloX = x + TILE_SIZE; haloY = y + TILE_SIZE;
        tile[tz][ty + TILE_SIZE][tx + TILE_SIZE] = (haloX < width && haloY < width && z < width) ? input[z * width * width + haloY * width + haloX] : 0.0f;
    }

    if (threadIdx.y < radius && threadIdx.z < radius) {
        int haloY = y - radius;
        int haloZ = z - radius;
        tile[threadIdx.z][threadIdx.y][tx] = (x < width && haloY >= 0 && haloZ >= 0) ? input[haloZ * width * width + haloY * width + x] : 0.0f;

        haloY = y + TILE_SIZE; haloZ = z - radius;
        tile[threadIdx.z][ty + TILE_SIZE][tx] = (x < width && haloY < width && haloZ >= 0) ? input[haloZ * width * width + haloY * width + x] : 0.0f;

        haloY = y - radius; haloZ = z + TILE_SIZE;
        tile[tz + TILE_SIZE][threadIdx.y][tx] = (x < width && haloY >= 0 && haloZ < width) ? input[haloZ * width * width + haloY * width + x] : 0.0f;

        haloY = y + TILE_SIZE; haloZ = z + TILE_SIZE;
        tile[tz + TILE_SIZE][ty + TILE_SIZE][tx] = (x < width && haloY < width && haloZ < width) ? input[haloZ * width * width + haloY * width + x] : 0.0f;
    }

    if (threadIdx.x < radius && threadIdx.z < radius) {
        int haloX = x - radius;
        int haloZ = z - radius;
        tile[threadIdx.z][ty][threadIdx.x] = (haloX >= 0 && y < width && haloZ >= 0) ? input[haloZ * width * width + y * width + haloX] : 0.0f;

        haloX = x + TILE_SIZE; haloZ = z - radius;
        tile[threadIdx.z][ty][tx + TILE_SIZE] = (haloX < width && y < width && haloZ >= 0) ? input[haloZ * width * width + y * width + haloX] : 0.0f;

        haloX = x - radius; haloZ = z + TILE_SIZE;
        tile[tz + TILE_SIZE][ty][threadIdx.x] = (haloX >= 0 && y < width && haloZ < width) ? input[haloZ * width * width + y * width + haloX] : 0.0f;

        haloX = x + TILE_SIZE; haloZ = z + TILE_SIZE;
        tile[tz + TILE_SIZE][ty][tx + TILE_SIZE] = (haloX < width && y < width && haloZ < width) ? input[haloZ * width * width + y * width + haloX] : 0.0f;
    }

    if (threadIdx.x < radius && threadIdx.y < radius && threadIdx.z < radius) {
        int coords[8][3] = {
            {x - radius, y - radius, z - radius},
            {x + TILE_SIZE, y - radius, z - radius},
            {x - radius, y + TILE_SIZE, z - radius},
            {x + TILE_SIZE, y + TILE_SIZE, z - radius},
            {x - radius, y - radius, z + TILE_SIZE},
            {x + TILE_SIZE, y - radius, z + TILE_SIZE},
            {x - radius, y + TILE_SIZE, z + TILE_SIZE},
            {x + TILE_SIZE, y + TILE_SIZE, z + TILE_SIZE}
        };

        int tileCoords[8][3] = {
            {threadIdx.x, threadIdx.y, threadIdx.z},
            {tx + TILE_SIZE, threadIdx.y, threadIdx.z},
            {threadIdx.x, ty + TILE_SIZE, threadIdx.z},
            {tx + TILE_SIZE, ty + TILE_SIZE, threadIdx.z},
            {threadIdx.x, threadIdx.y, tz + TILE_SIZE},
            {tx + TILE_SIZE, threadIdx.y, tz + TILE_SIZE},
            {threadIdx.x, ty + TILE_SIZE, tz + TILE_SIZE},
            {tx + TILE_SIZE, ty + TILE_SIZE, tz + TILE_SIZE}
        };

        for (int i = 0; i < 8; i++) {
            int haloX = coords[i][0];
            int haloY = coords[i][1];
            int haloZ = coords[i][2];
            tile[tileCoords[i][2]][tileCoords[i][1]][tileCoords[i][0]] =
                (haloX >= 0 && haloX < width && haloY >= 0 && haloY < width && haloZ >= 0 && haloZ < width)
                ? input[haloZ * width * width + haloY * width + haloX]
                : 0.0f;
        }
    }

    __syncthreads();

    if (x < width && y < width && z < width) {
        float sum = 0.0f;
        int idx = 0;
#pragma unroll
        for (int kz = 0; kz < 3; kz++) {
#pragma unroll
            for (int ky = 0; ky < 3; ky++) {
#pragma unroll
                for (int kx = 0; kx < 3; kx++) {
                    sum += tile[tz + kz - 1][ty + ky - 1][tx + kx - 1] * d_filter3D[idx++];
                }
            }
        }
        output[z * width * width + y * width + x] = sum;
    }
}

    __host__ void check_result(const float* w_ref, const float* w_opt, int size) {
    double maxdiff = 0.0;
    int numdiffs = 0;
    
    for (int i = 0; i < size; i++) {
        double this_diff = w_ref[i] - w_opt[i];
        if (std::fabs(this_diff) > THRESHOLD) {
        numdiffs++;
        if (std::fabs(this_diff) > maxdiff) {
            maxdiff = std::fabs(this_diff);
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

    double rtclock() {
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) {
        cout << "Error return from gettimeofday: " << stat << "\n";
    }
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
    }

    float launch_kernel2D_opt(int tileSize, float* d_input, float* d_output, 
                            cudaEvent_t start, cudaEvent_t stop) {
    dim3 block(tileSize, tileSize);
    dim3 grid((N + tileSize - 1) / tileSize, (N + tileSize - 1) / tileSize);
    
    cudaCheckError(cudaEventRecord(start));
    
    switch(tileSize) {
        case 2:
        kernel2D_opt_templated<2><<<grid, block>>>(d_input, d_output, N, FILTER_SIZE);
        break;
        case 4:
        kernel2D_opt_templated<4><<<grid, block>>>(d_input, d_output, N, FILTER_SIZE);
        break;
        case 8:
        kernel2D_opt_templated<8><<<grid, block>>>(d_input, d_output, N, FILTER_SIZE);
        break;
        case 16:
        kernel2D_opt_templated<16><<<grid, block>>>(d_input, d_output, N, FILTER_SIZE);
        break;
    }
    
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));
    
    float kernel_time;
    cudaCheckError(cudaEventElapsedTime(&kernel_time, start, stop));
    return kernel_time;
    }

float launch_kernel3D_opt(int tileSize, float* d_input, float* d_output,
                        cudaEvent_t start, cudaEvent_t stop) {
    // Reject tile sizes that don't fit in shared memory to avoid "empty" timings
    cudaDeviceProp prop;
    cudaCheckError(cudaGetDeviceProperties(&prop, 0));
    size_t sharedDim = tileSize + 2 * FILTER_RADIUS;
    size_t sharedBytes = sharedDim * sharedDim * sharedDim * sizeof(float);
    if (sharedBytes > static_cast<size_t>(prop.sharedMemPerBlock)) {
        cerr << "Tile size " << tileSize << "^3 requires " << sharedBytes
             << " bytes > sharedMemPerBlock (" << prop.sharedMemPerBlock
             << "). Skipping.\n";
        return 0.0f;
    }

    dim3 block(tileSize, tileSize, tileSize);
    dim3 grid((N + tileSize - 1) / tileSize, 
                (N + tileSize - 1) / tileSize,
                (N + tileSize - 1) / tileSize);
    
    cudaCheckError(cudaEventRecord(start));
    
    switch(tileSize) {
        case 2:
        kernel3D_opt_templated<2><<<grid, block>>>(d_input, d_output, N, FILTER_SIZE);
        break;
        case 4:
        kernel3D_opt_templated<4><<<grid, block>>>(d_input, d_output, N, FILTER_SIZE);
        break;
        case 8:
        kernel3D_opt_templated<8><<<grid, block>>>(d_input, d_output, N, FILTER_SIZE);
        break;
        case 16:
        kernel3D_opt_templated<16><<<grid, block>>>(d_input, d_output, N, FILTER_SIZE);
        break;
    }
    
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));
    
    float kernel_time;
    cudaCheckError(cudaEventElapsedTime(&kernel_time, start, stop));
    return kernel_time;
    }

    int main()
    {
        // Allocate host memory for 2D
        int size2D = N * N * sizeof(float);
        int filterSize2D = FILTER_SIZE * FILTER_SIZE * sizeof(float);

        float *h_input2D = (float *)malloc(size2D);
        float *h_filter2D = (float *)malloc(filterSize2D);
        float *h_output2D_basic = (float *)malloc(size2D);
        float *h_output2D_opt = (float *)malloc(size2D);

        // Initialize 2D input and filter
        srand(42);
        for (int i = 0; i < N * N; i++)
        {
            h_input2D[i] = (float)(rand() % 100) / 10.0f;
        }

        float filterSum = 0.0f;
        for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++)
        {
            h_filter2D[i] = 1.0f;
            filterSum += h_filter2D[i];
        }
        for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++)
        {
            h_filter2D[i] /= filterSum;
        }

        // Allocate device memory for 2D
        float *d_input2D, *d_filter2D_global, *d_output2D; // CHANGED
        cudaCheckError(cudaMalloc(&d_input2D, size2D));
        cudaCheckError(cudaMalloc(&d_filter2D_global, filterSize2D)); // CHANGED
        cudaCheckError(cudaMalloc(&d_output2D, size2D));

        cudaCheckError(cudaMemcpy(d_input2D, h_input2D, size2D, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_filter2D_global, h_filter2D, filterSize2D, cudaMemcpyHostToDevice)); // CHANGED
        cudaCheckError(cudaMemcpyToSymbol(d_filter2D, h_filter2D, filterSize2D));                        // Now correct - d_filter2D refers to __constant__

        cudaEvent_t start, stop;
        cudaCheckError(cudaEventCreate(&start));
        cudaCheckError(cudaEventCreate(&stop));

        // 2D Basic kernel
        dim3 block2D(16, 16);
        dim3 grid2D((N + block2D.x - 1) / block2D.x, (N + block2D.y - 1) / block2D.y);

        cudaCheckError(cudaEventRecord(start));
        kernel2D_basic<<<grid2D, block2D>>>(d_input2D, d_filter2D_global, d_output2D, N, FILTER_SIZE); // CHANGED
        cudaCheckError(cudaEventRecord(stop));
        cudaCheckError(cudaEventSynchronize(stop));

        float kernel_time;
        cudaCheckError(cudaEventElapsedTime(&kernel_time, start, stop));
        cout << "Kernel2D_basic time (ms): " << kernel_time << "\n\n";

        cudaCheckError(cudaMemcpy(h_output2D_basic, d_output2D, size2D, cudaMemcpyDeviceToHost));

        // Test different tile sizes for 2D
        cout << "Testing 2D tile sizes:\n";
        cout << "---------------------\n";
        float bestTime2D = 1e9;
        int bestTileSize2D = TILE_SIZES_2D[0];

        for (int i = 0; i < NUM_TILE_SIZES_2D; i++)
        {
            int tileSize = TILE_SIZES_2D[i];

            // Warm up
            launch_kernel2D_opt(tileSize, d_input2D, d_output2D, start, stop);

            // Measure multiple runs
            const int NUM_RUNS = 5;
            float totalTime = 0.0f;
            for (int run = 0; run < NUM_RUNS; run++)
            {
                totalTime += launch_kernel2D_opt(tileSize, d_input2D, d_output2D, start, stop);
            }
            float avgTime = totalTime / NUM_RUNS;

            // Check correctness
            cudaCheckError(cudaMemcpy(h_output2D_opt, d_output2D, size2D, cudaMemcpyDeviceToHost));
            cout << "Tile size " << tileSize << "x" << tileSize << ": " << avgTime << " ms - ";
            check_result(h_output2D_basic, h_output2D_opt, N * N);

            if (avgTime < bestTime2D)
            {
                bestTime2D = avgTime;
                bestTileSize2D = tileSize;
            }
        }

        cout << "\nBest 2D tile size: " << bestTileSize2D << "x" << bestTileSize2D
            << " with time: " << bestTime2D << " ms\n";
        cout << "Speedup over basic: " << kernel_time / bestTime2D << "x\n\n";

    // === 3D Convolution ===
    // Use size_t to avoid overflow for larger N (e.g., 2^10)
    size_t size3D = static_cast<size_t>(N) * N * N * sizeof(float);
    size_t filterSize3D = FILTER_SIZE * FILTER_SIZE * FILTER_SIZE * sizeof(float);

        float *h_input3D = (float *)malloc(size3D);
        float *h_filter3D = (float *)malloc(filterSize3D);
        float *h_output3D_basic = (float *)malloc(size3D);
        float *h_output3D_opt = (float *)malloc(size3D);

        // Initialize 3D input and filter
        for (int i = 0; i < N * N * N; i++)
        {
            h_input3D[i] = (float)(rand() % 100) / 10.0f;
        }

        filterSum = 0.0f;
        for (int i = 0; i < FILTER_SIZE * FILTER_SIZE * FILTER_SIZE; i++)
        {
            h_filter3D[i] = 1.0f;
            filterSum += h_filter3D[i];
        }
        for (int i = 0; i < FILTER_SIZE * FILTER_SIZE * FILTER_SIZE; i++)
        {
            h_filter3D[i] /= filterSum;
        }

        // Allocate device memory for 3D
        float *d_input3D, *d_filter3D_global, *d_output3D; // CHANGED
        cudaCheckError(cudaMalloc(&d_input3D, size3D));
        cudaCheckError(cudaMalloc(&d_filter3D_global, filterSize3D)); // CHANGED
        cudaCheckError(cudaMalloc(&d_output3D, size3D));

        cudaCheckError(cudaMemcpy(d_input3D, h_input3D, size3D, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_filter3D_global, h_filter3D, filterSize3D, cudaMemcpyHostToDevice)); // CHANGED
        cudaCheckError(cudaMemcpyToSymbol(d_filter3D, h_filter3D, filterSize3D));                        // Now correct

        // 3D Basic kernel
        dim3 block3D(8, 8, 8);
        dim3 grid3D((N + block3D.x - 1) / block3D.x,
                    (N + block3D.y - 1) / block3D.y,
                    (N + block3D.z - 1) / block3D.z);

        cudaCheckError(cudaEventRecord(start));
        kernel3D_basic<<<grid3D, block3D>>>(d_input3D, d_filter3D_global, d_output3D, N, FILTER_SIZE); // CHANGED
        cudaCheckError(cudaEventRecord(stop));
        cudaCheckError(cudaEventSynchronize(stop));

        cudaCheckError(cudaEventElapsedTime(&kernel_time, start, stop));
        cout << "Kernel3D_basic time (ms): " << kernel_time << "\n\n";

        cudaCheckError(cudaMemcpy(h_output3D_basic, d_output3D, size3D, cudaMemcpyDeviceToHost));

        // Test different tile sizes for 3D
        cout << "Testing 3D tile sizes:\n";
        cout << "---------------------\n";
        float bestTime3D = 1e9;
        int bestTileSize3D = TILE_SIZES_3D[0];

        for (int i = 0; i < NUM_TILE_SIZES_3D; i++)
        {
            int tileSize = TILE_SIZES_3D[i];

            // Measure multiple runs
            const int NUM_RUNS = 1;
            float totalTime = 0.0f;
            for (int run = 0; run < NUM_RUNS; run++)
            {
                totalTime += launch_kernel3D_opt(tileSize, d_input3D, d_output3D, start, stop);
            }
            float avgTime = totalTime / NUM_RUNS;

            // Check correctness
            cudaCheckError(cudaMemcpy(h_output3D_opt, d_output3D, size3D, cudaMemcpyDeviceToHost));
            cout << "Tile size " << tileSize << "x" << tileSize << "x" << tileSize
                << ": " << avgTime << " ms - ";
            check_result(h_output3D_basic, h_output3D_opt, N * N * N);

            if (avgTime < bestTime3D)
            {
                bestTime3D = avgTime;
                bestTileSize3D = tileSize;
            }
        }

        cout << "\nBest 3D tile size: " << bestTileSize3D << "x" << bestTileSize3D
            << "x" << bestTileSize3D << " with time: " << bestTime3D << " ms\n";
        cout << "Speedup over basic: " << kernel_time / bestTime3D << "x\n";

        // Free memory
        free(h_input2D);
        free(h_filter2D);
        free(h_output2D_basic);
        free(h_output2D_opt);
        free(h_input3D);
        free(h_filter3D);
        free(h_output3D_basic);
        free(h_output3D_opt);

        cudaFree(d_input2D);
        cudaFree(d_filter2D_global); // CHANGED
        cudaFree(d_output2D);
        cudaFree(d_input3D);
        cudaFree(d_filter3D_global); // CHANGED
        cudaFree(d_output3D);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return EXIT_SUCCESS;
    }
