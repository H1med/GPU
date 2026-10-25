#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU(void) {
    printf("Hello World from GPU!\n");
}

int main(void) {
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
