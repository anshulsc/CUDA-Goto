#include "common.h"
#include "timer.h"


void rgb2gray_cpu(float* r, float* g, float* b, float* gray, int N){
    for(unsigned int i = 0; i < N; ++i){
        gray[i] = 0.21f*r[i] + 0.71f*g[i] + 0.07f*b[i];
    }
}


__global__ void rgb2gray_kernel(unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* gray, unsigned int width, unsigned int height){
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col < width && row < height){
        unsigned int idx = row * width + col;
        gray[idx] = 0.21f*r[idx] + 0.71f*g[idx] + 0.07f*b[idx];
    }
}

void rgb2gray_gpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height){

    Timer timer;
    // Allocate GPU memory 
    startTime(&timer);
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaMalloc((void**)&red_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**)&green_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**)&blue_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**)&gray_d, width*height*sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Memory Allocation Time", YELLOW);

    // Copy to the GPU
    startTime(&timer); 
    cudaMemcpy(red_d, red, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Memory Copy Time", YELLOW);




    // Run the GPU Code
    // call a GPU Kernel function (launch a grid of threads)
    Timer timer;
    startTime(&timer);
    dim3 numThreadsPerBlock(32,32);
    dim3 numBlock((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    rgb2gray_kernel<<< numBlocks, numThreadsPerBlock >>>(red_d, green_d, blue_d, gray_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Kernel Time", YELLOW);

    // Copy back to the CPU
    startTime(&timer);
    cudaMemcpy(gray, gray_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Memory Copy Back Time", YELLOW);

    // Dealliocate GPU Memory
    startTime(&timer);
    cudaFree(red_d);    
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Memory Deallocation Time", YELLOW);

}

int main(int argc, char**argv){

    cudaDeviceSynchronize();

    Timer timer;
    unsigned int width = (argc > 1)?(atoi(argv[1])):(1 << 10);
    unsigned int height = (argc > 2)?(atoi(argv[2])):(1 << 10);
    unsigned int N = width * height;
    unsigned char* red = (unsigned char*) malloc(N*sizeof(unsigned char));
    unsigned char* green = (unsigned char*) malloc(N*sizeof(unsigned char));
    unsigned char* blue = (unsigned char*) malloc(N*sizeof(unsigned char));
    unsigned char* gray = (unsigned char*) malloc(N*sizeof(unsigned char));
    for(unsigned int i = 0; i < N; ++i){
        red[i] = rand();
        green[i] = rand();
        blue[i] = rand();
    }

    // CPU
    startTime(&timer);
    rgb2gray_cpu(red, green, blue, gray, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU Time", YELLOW);

    // GPU
    rgb2gray_gpu(red, green, blue, gray, width, height);

    free(red);
    free(green);
    free(blue);
    free(gray);

    return 0;
}