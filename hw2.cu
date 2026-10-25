// Mohamed Abdul Hussein 108021237600

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define precision 1e-5f

void createMatrix(float *M, int n)
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

__global__ void matrixAddGPU(const float *A, const float *B, float *C, int rows, int cols)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

int checkResult(const float *cpuRef, const float *gpuRes, int n)
{
    for (int i = 0; i < n; ++i) {
        if (fabsf(cpuRef[i] - gpuRes[i]) > precision) {
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

float timeGPU(float *d_A, float *d_B, float *d_C, int rows, int cols, int bx, int by, int iter)
{
    dim3 block(bx, by, 1);
    dim3 grid((cols + bx - 1) / bx, (rows + by - 1) / by, 1);
    cudaEvent_t start, stop;
    float ms = 0.0f;
    int i;

    matrixAddGPU<<<grid, block>>>(d_A, d_B, d_C, rows, cols);
    cudaDeviceSynchronize();

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (i = 0; i < iter; ++i)
        matrixAddGPU<<<grid, block>>>(d_A, d_B, d_C, rows, cols);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / (float)iter;
}

void runCase(int rows, int cols, int iterCpu, int iterGpu, int bx, int by)
{
    int n = rows * cols;
    size_t bytes = (size_t)n * sizeof(float);
    float *A, *B, *Ccpu, *Cgpu;
    float *d_A, *d_B, *d_C;
    double cpuMs, speedup;
    float gpuMs;
    int ok;

    A = (float*)malloc(bytes);
    B = (float*)malloc(bytes);
    Ccpu = (float*)malloc(bytes);
    Cgpu = (float*)malloc(bytes);

    createMatrix(A, n);
    createMatrix(B, n);

    cpuMs = timeCPU(A, B, Ccpu, rows, cols, iterCpu);

    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    gpuMs = timeGPU(d_A, d_B, d_C, rows, cols, bx, by, iterGpu);

    cudaMemcpy(Cgpu, d_C, bytes, cudaMemcpyDeviceToHost);

    ok = checkResult(Ccpu, Cgpu, n);
    speedup = (gpuMs > 0.0f) ? cpuMs / (double)gpuMs : 0.0;

    printf("%5d x %-5d  %2d x %-2d  %12.6f  %12.6f  %8.2f  %s\n",
           rows, cols, bx, by, cpuMs, gpuMs, speedup, ok ? "OK" : "FAIL");

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
    int sizes[5][2] = {
        {10, 10},
        {100, 100},
        {1000, 1000},
        {500, 2000},
        {100, 10000}
    };
    int itersCpu[5] = {1000000, 100000, 200, 200, 200};
    int itersGpu[5] = {100000, 10000, 200, 200, 200};
    int i;

    srand((unsigned)time(NULL));

    printf("Matrix addition: CPU vs GPU (host<->device transfers excluded)\n\n");

    printf("%-16s %-12s %14s %14s %10s  %s\n",
           "Matrix", "Block", "CPU [ms]", "GPU [ms]", "Speedup", "Check");

    for (i = 0; i < 5; ++i)
        runCase(sizes[i][0], sizes[i][1], itersCpu[i], itersGpu[i], 16, 16);

    printf("\nExecution configurations for 100 x 10000:\n\n");

    printf("%-16s %-12s %14s %14s %10s  %s\n",
           "Matrix", "Block", "CPU [ms]", "GPU [ms]", "Speedup", "Check");

    runCase(100, 10000, 200, 200, 16, 16);
    runCase(100, 10000, 200, 200, 16, 32);
    runCase(100, 10000, 200, 200, 32, 16);

    return 0;
}
