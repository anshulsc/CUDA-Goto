#include "timer.h"

void vecadd_cpu(float* x, float* y, float* z , int N){
    for(unsigned int i = 0; i < N: ++i){
        z[i] = x[i] + y[i];
    }
}

__global__ void vecadd_kernel(float* x, float* y, float* z, int N){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        z[i] = x[i] + y[i];
    }
}

void vecadd_gpu(float* x, float* y , float* z, int N){

    // Allocate GPU memory 
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d, N*sizeof(float));
    cudaMalloc((void**)&y_d, N*sizeof(float));
    cudaMalloc((void**)&z_d, N*sizeof(float));

    // Copy to the GPU 
    cudeMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudeMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Run the GPU Code
    // call a GPU Kernel function (launch a grid of threads)
    Timer timer;
    startTime(&timer);
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    vecadd_kernel<<< numBlocks, numThreadsPerBlock >>>(x_d, y_d, z_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Kernel Time", YELLOW);


    // Copy back to the CPU
    cudeMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Dealliocate GPU Memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);


}

int main(int argc, char**argv){

    cudaDeviceSynchronize();

    Timer timer;
    unsigned int N = (argc > 1)?(atoi(argv[1])):(1 << 25);
    float* x = (float*) malloc(N*sizeof(float));
    float* y = (float*) malloc(N*sizeof(float));
    float* z = (float*) malloc(N*sizeof(float));
    for(unsigned int i = 0; i < N; ++i){
        x[i] = rand();
        y[i] = rand();
    }

    //vector addition on CPU
    startTime(&timer);
    vecadd_cpu(x, y, z, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU Time", CYAN);

    //vector addition on GPU
    startTime(&timer);
    vecadd_gpu(x, y, z, N);
    stopTime(&timer);
    printElapsedTime(timer, "GPU Time", CYAN);

    // Free the memory
    free(x);
    free(y);
    free(z);

    return 0;

}