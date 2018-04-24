#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

__global__ void getmaxcu(){

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

    //allocating on the device
    unsigned int max;
    unsigned int* numbers_device, max_device;
    cudaError_t error = cudaMalloc((void**)&numbers_device, size * sizeof(unsigned int));

    //error handling
    if(error != cudaSuccess){ // print the CUDA error message and exit printf("CUDA error: %s\n",
      cudaGetErrorString(error);
      exit(-1);
    }

    error = cudaMalloc((void**)&max_device, sizeof(unsigned int))

    if(error != cudaSuccess){ // print the CUDA error message and exit printf("CUDA error: %s\n",
      cudaGetErrorString(error);
      exit(-1);
    }

    cudaMemcpy(num_device, numbers, size * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //lauch pre-defined kernel code
   
    printf(" The maximum number in the array is: %u\n", 
           getmax(numbers, size));

    free(numbers);
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
