// Mohamed Abdul Hussein 108021237600

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define ROWS 1000
#define COLS 1000
#define BLOCK_MIN 2
#define BLOCK_MAX 32
#define BLOCK_STEP 2
#define BLOCK_CONFIGS 16
#define SWEEP_ITER 1000
#define SWEEP_CPU_ITER 100

#define DIV_N (1 << 18)
#define DIV_WORK 100
#define DIV_BLOCK 128
#define DIV_GROUPS 32
#define DIV_TIME_ITER 200


void initialData(float *M, int n)
{
    for (int i = 0; i < n; ++i)
        M[i] = (float)rand() / (float)RAND_MAX;
}

void matrixAddCPU(const float *A, const float *B, float *C, int rows, int cols)
{
    int n = rows * cols;

    for (int i = 0; i < n; ++i)
        C[i] = A[i] + B[i];
}

void vectorAddDivergentCPU(const float *A, const float *B, float *C, int n, int groups)
{
    for (int idx = 0; idx < n; ++idx) {
        int lane = idx % 32;
        float sum = A[idx] + B[idx];

        for (int i = 0; i < groups; ++i) {
            if (lane >= (int)(32.0f * i / (float)groups) &&
                lane < (int)(32.0f * (i + 1) / (float)groups)) {
                for (int k = 0; k < DIV_WORK; ++k)
                    sum = sum * B[idx] + (float)(i + 1);
            }
        }

        C[idx] = sum;
    }
}

__global__ void matrixAddGPU(const float *A, const float *B, float *C, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void vectorAddDivergentGPU(const float *A, const float *B, float *C, int n, int groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int lane = threadIdx.x % 32;
        float sum = A[idx] + B[idx];

        for (int i = 0; i < groups; ++i) {
            if (lane >= (int)(32.0f * i / (float)groups) &&
                lane < (int)(32.0f * (i + 1) / (float)groups)) {
                for (int k = 0; k < DIV_WORK; ++k)
                    sum = sum * B[idx] + (float)(i + 1);
            }
        }

        C[idx] = sum;
    }
}

int checkResult(const float *cpuRef, const float *gpuRes, int n, double epsilon)
{
    for (int i = 0; i < n; ++i) {
        if (fabsf(cpuRef[i] - gpuRes[i]) > epsilon) {
            printf("Mismatch at index %d: CPU=%f GPU=%f\n", i, cpuRef[i], gpuRes[i]);
            return 0;
        }
    }

    return 1;
}

double timeCPU(const float *A, const float *B, float *C, int rows, int cols, int iter)
{
    clock_t start, stop;

    start = clock();

    for (int i = 0; i < iter; ++i)
        matrixAddCPU(A, B, C, rows, cols);

    stop = clock();

    return 1000.0 * (double)(stop - start) / (double)CLOCKS_PER_SEC / (double)iter;
}

float timeMatrixGPU(float *d_A, float *d_B, float *d_C, int rows, int cols, int bx, int by, int iter)
{
    dim3 block(bx, by);
    dim3 grid((cols + bx - 1) / bx, (rows + by - 1) / by);
    cudaEvent_t start, stop;
    float ms = 0.0f;

    matrixAddGPU<<<grid, block>>>(d_A, d_B, d_C, rows, cols);
    cudaDeviceSynchronize();

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < iter; ++i)
        matrixAddGPU<<<grid, block>>>(d_A, d_B, d_C, rows, cols);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / (float)iter;
}

float timeVectorGPU(float *d_A, float *d_B, float *d_C, int n, int groups, int iter)
{
    int block = DIV_BLOCK;
    int grid = (n + block - 1) / block;
    cudaEvent_t start, stop;
    float ms = 0.0f;

    vectorAddDivergentGPU<<<grid, block>>>(d_A, d_B, d_C, n, groups);
    cudaDeviceSynchronize();

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < iter; ++i)
        vectorAddDivergentGPU<<<grid, block>>>(d_A, d_B, d_C, n, groups);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / (float)iter;
}

int occupancyPercent(int blockSize)
{
    int blocksPerSM, maxThreads, warpSize;
    int warpsPerBlock = (blockSize + 31) / 32;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, matrixAddGPU, blockSize, 0);
    cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, 0);

    return (int)(100.0 * blocksPerSM * warpsPerBlock * warpSize / maxThreads);
}

void runMatrixSweep(void)
{
    int rows = ROWS;
    int cols = COLS;
    int n = rows * cols;
    int bytes = n * sizeof(float);
    float *A, *B, *Ccpu, *Cgpu;
    float *d_A, *d_B, *d_C;
    float times[BLOCK_CONFIGS][BLOCK_CONFIGS];
    int occ[BLOCK_CONFIGS][BLOCK_CONFIGS];
    int bestIx = 0;
    int bestIy = 0;
    float bestMs = 1.0e30f;
    double cpuMs;

    A = (float*)malloc(bytes);
    B = (float*)malloc(bytes);
    Ccpu = (float*)malloc(bytes);
    Cgpu = (float*)malloc(bytes);

    initialData(A, n);
    initialData(B, n);

    cpuMs = timeCPU(A, B, Ccpu, rows, cols, SWEEP_CPU_ITER);

    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    printf("Matrix addition: CPU vs GPU, %d x %d\n\n", rows, cols);
    printf("CPU time: %.6f ms\n", cpuMs);

    for (int iy = 0; iy < BLOCK_CONFIGS; ++iy) {
        int by = BLOCK_MIN + iy * BLOCK_STEP;

        for (int ix = 0; ix < BLOCK_CONFIGS; ++ix) {
            int bx = BLOCK_MIN + ix * BLOCK_STEP;
            float ms;

            ms = timeMatrixGPU(d_A, d_B, d_C, rows, cols, bx, by, SWEEP_ITER);

            times[iy][ix] = ms;
            occ[iy][ix] = occupancyPercent(bx * by);

            if (ms < bestMs) {
                bestMs = ms;
                bestIx = ix;
                bestIy = iy;
            }
        }
    }

    printf("\nExecution time (ms), rows = blockDim.y, columns = blockDim.x\n\n");
    printf("%7s", "by\\bx");

    for (int ix = 0; ix < BLOCK_CONFIGS; ++ix)
        printf("%9d", BLOCK_MIN + ix * BLOCK_STEP);

    printf("\n");

    for (int iy = 0; iy < BLOCK_CONFIGS; ++iy) {
        printf("%7d", BLOCK_MIN + iy * BLOCK_STEP);

        for (int ix = 0; ix < BLOCK_CONFIGS; ++ix)
            printf("%9.6f", times[iy][ix]);

        printf("\n");
    }

    printf("\nOccupancy (%%), rows = blockDim.y, columns = blockDim.x\n\n");
    printf("%7s", "by\\bx");

    for (int ix = 0; ix < BLOCK_CONFIGS; ++ix)
        printf("%5d", BLOCK_MIN + ix * BLOCK_STEP);

    printf("\n");

    for (int iy = 0; iy < BLOCK_CONFIGS; ++iy) {
        printf("%7d", BLOCK_MIN + iy * BLOCK_STEP);

        for (int ix = 0; ix < BLOCK_CONFIGS; ++ix)
            printf("%5d", occ[iy][ix]);

        printf("\n");
    }

    dim3 block(BLOCK_MIN + bestIx * BLOCK_STEP, BLOCK_MIN + bestIy * BLOCK_STEP);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    int ok;

    matrixAddGPU<<<grid, block>>>(d_A, d_B, d_C, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(Cgpu, d_C, bytes, cudaMemcpyDeviceToHost);
    ok = checkResult(Ccpu, Cgpu, n, 1.0E-8);

    printf("\nBest configuration: %d x %d threads\n",
           BLOCK_MIN + bestIx * BLOCK_STEP, BLOCK_MIN + bestIy * BLOCK_STEP);
    printf("Time: %.6f ms, Occupancy: %d %%, Check: %s\n",
           bestMs, occ[bestIy][bestIx], ok ? "OK" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(Ccpu);
    free(Cgpu);
}

void runDivergenceCase(void)
{
    int n = DIV_N;
    int bytes = n * sizeof(float);
    float *A, *B, *Ccpu, *Cgpu;
    float *d_A, *d_B, *d_C;
    float firstMs = 0.0f;
    int okAll = 1;

    A = (float*)malloc(bytes);
    B = (float*)malloc(bytes);
    Ccpu = (float*)malloc(bytes);
    Cgpu = (float*)malloc(bytes);

    initialData(A, n);
    initialData(B, n);

    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    printf("\nWarp control divergence: 1D vector addition\n\n");
    printf("Vector: %d, work per thread: %d, block: %d x 1\n\n", n, DIV_WORK, DIV_BLOCK);
    printf("%-8s %12s %10s  %s\n", "Groups", "Time", "Slowdown", "Check");

    for (int groups = 1; groups <= DIV_GROUPS; ++groups) {
        float ms;
        int ok;

        vectorAddDivergentCPU(A, B, Ccpu, n, groups);

        ms = timeVectorGPU(d_A, d_B, d_C, n, groups, DIV_TIME_ITER);

        cudaMemcpy(Cgpu, d_C, bytes, cudaMemcpyDeviceToHost);

        ok = checkResult(Ccpu, Cgpu, n, 1.0E-2);

        if (!ok)
            okAll = 0;

        if (groups == 1)
            firstMs = ms;

        printf("%-8d %12.6f %10.2f  %s\n", groups, ms, ms / firstMs, ok ? "OK" : "FAIL");
    }

    printf("\nAll checks: %s\n", okAll ? "OK" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(Ccpu);
    free(Cgpu);
}

int main(void)
{
    srand((unsigned)time(NULL));

    runMatrixSweep();
    runDivergenceCase();

    return 0;
}
