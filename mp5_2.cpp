// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here
  
  __shared__ float sharedMemory[2*BLOCK_SIZE];

  int bx = blockIdx.x; int tx = threadIdx.x;
  int x_in = bx * BLOCK_SIZE*2 + tx;
  
  // give 0 values to elements beyond the matrix range load both left and right block elements
  if (x_in >= 0 && x_in < len)
  {
    sharedMemory[tx] = input[x_in];
    sharedMemory[tx + BLOCK_SIZE] = input[x_in + BLOCK_SIZE];
    //printf("x_in: %i, val: %f \n", x_in, sharedMemory[x_in]);
    //printf("x_in: %i, val: %f \n", x_in + BLOCK_SIZE, sharedMemory[tx + BLOCK_SIZE]);
  }
  else sharedMemory[tx] = 0;
  __syncthreads();
  
  for (int stride = 1; stride <= BLOCK_SIZE; stride = stride*2)
  {
    int index = (threadIdx.x+1)*stride*2 - 1; 
    if(index < 2*BLOCK_SIZE) sharedMemory[index] += sharedMemory[index-stride];
    
    __syncthreads();
    //if (tx == 0) printf("tx: %i, val: %f \n", tx, sharedMemory[index]); // active thread (0) printout
  }
  
  for (int stride = BLOCK_SIZE/2; stride > 0;stride /= 2)
  {
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index + stride< 2*BLOCK_SIZE) 
    { 
      sharedMemory[index + stride] += sharedMemory[index]; 
    }
    __syncthreads();
  }
  
  if (x_in < len) output[x_in] = sharedMemory[tx];
  if (x_in+blockDim.x < len) output[x_in+blockDim.x] = sharedMemory[threadIdx.x+blockDim.x];

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(numElements/(2*BLOCK_SIZE), 1, 1);
  if (numElements%(2*BLOCK_SIZE)) dimGrid.x++;
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  float *add_value_p;
  add_value_p = (float *)malloc(dimGrid.x * sizeof(float));
  add_value_p[0] = 0;
  for (int i = 1; i < dimGrid.x; i++){
    add_value_p[i] = add_value_p[i-1] + hostOutput[2*BLOCK_SIZE*i-1];
    printf("%i %f \n",i,add_value_p[i]);
  } 
  
  for (int i = 0; i < numElements; i++) hostOutput[i] += add_value_p[i/(2*BLOCK_SIZE)];
  
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
