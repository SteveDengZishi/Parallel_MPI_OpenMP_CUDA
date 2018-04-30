#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

__global__ void getmaxcu(unsigned int* numbers_device, unsigned int* max_device, unsigned int size){

    __device__ __shared__ unsigned int shared_num[1024];

    //copy from device global memory to device shared memory
    shared_num[threadIdx.x] = numbers_device[blockDim.x * blockIdx.x + threadIdx.x];
    __syncthreads();

    //use reduction to find max
    unsigned int tid=threadIdx.x;
    unsigned int i;
    for(i=blockDim.x>>1;i>0;i>>=1){
      __syncthreads();
      if(tid<i){
        shared_num[tid]=max(shared_num[tid],shared_num[tid+i]);
      }
    }
    __syncthreads();
    //shared_num[0] is the maximum by now in each blocks
    if(threadIdx.x==0){
      atomicMax(max_device, shared_num[0]);
    }
}

int main(int argc, char *argv[])
{
    unsigned int size = 0;  // The size of the array
    unsigned int i;  // loop index
    unsigned int * numbers; //pointer to the array
    
    if(argc !=2)
    {
       printf("usage: maxseq num\n");
       printf("num = size of the array\n");
       exit(1);
    }
   
    size = atol(argv[1]);
    numbers = (unsigned int *)malloc(size * sizeof(unsigned int));

    if( !numbers )
    {
       printf("Unable to allocate mem for an array of size %u\n", size);
       exit(1);
    }

    srand(time(NULL)); // setting a seed for the random number generator
    // Fill-up the array with random numbers from 0 to size-1 
    for( i = 0; i < size; i++) numbers[i] = rand()  % size;  

    /*
    //checking and printing device properties
    int device;
    cudaDeviceProp cuda_properties;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&cuda_properties,device);
    printf("Device Properties for %s\n",cuda_properties.name);
    printf("================================================\n");
    printf("Total Global Memory Size is %u\n", cuda_properties.totalGlobalMem);
    printf("Shared Memory per block is %u\n", cuda_properties.sharedMemPerBlock);
    printf("Warp Size is %d and register per block is %d\n", cuda_properties.warpSize, cuda_properties.regsPerBlock);
    printf("Max threads per block is %d\n", cuda_properties.maxThreadsPerBlock);
    printf("================================================\n");
    */
    
    //allocating on the device
    unsigned int max=0;
    unsigned int * numbers_device;
    unsigned int * max_device;
    cudaError_t error = cudaMalloc((void**)&numbers_device, size * sizeof(unsigned int));

    //error handling
    if(error != cudaSuccess){ // print the CUDA error message and exit printf("CUDA error: %s\n",
      cudaGetErrorString(error);
      exit(-1);
    }

    error = cudaMalloc((void**)&max_device, sizeof(unsigned int));

    if(error != cudaSuccess){ // print the CUDA error message and exit printf("CUDA error: %s\n",
      cudaGetErrorString(error);
      exit(-1);
    }

    cudaMemcpy(numbers_device, numbers, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(max_device, &max, sizeof(unsigned int), cudaMemcpyHostToDevice);

    //lauch pre-defined kernel code
    int block_size=1024;
    int block_num=ceil((double)size/(double)block_size);

    //invoke kernel
    getmaxcu<<<block_num,block_size>>>(numbers_device,max_device,size);
    cudaDeviceSynchronize();
    //copy max_device back to host
    cudaMemcpy(&max, max_device, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("The maximum number in the array is: %u\n", max);

    //memory management
    free(numbers);
    cudaFree(numbers_device);
    cudaFree(max_device);
    exit(0); 
}


/*
   input: pointer to an array of long int
          number of elements in the array
   output: the maximum number of the array

unsigned int getmax(unsigned int num[], unsigned int size)
{
  unsigned int i;
  unsigned int max = num[0];

  for(i = 1; i < size; i++)
	if(num[i] > max)
	   max = num[i];

  return( max );

}
*/
