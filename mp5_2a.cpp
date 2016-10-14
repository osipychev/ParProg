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

// Addition kernel, for final hiearchical step
__global__ void vecAddKernel(float *in_d, float *out_d, int len) {
  
  __shared__ float sharedMemory[2*BLOCK_SIZE];
  
  int bx = blockIdx.x; int tx = threadIdx.x;
  int index = bx * blockDim.x + tx; // convert thread id into vector index
  
  if (index<len) sharedMemory[tx] = in_d[index]; // copy values to shared memory
  
  float temp = 0.0; //% find the value of the last element of the previos block
  if (bx > 0) for (int i = 1; i <= bx; i++) temp += in_d[i * 2 * BLOCK_SIZE - 1];
  //printf("%f \n", temp);
   
  sharedMemory[tx] += temp; // do addition for threads which require hierarchical addition

  if (index<len) out_d[index] = sharedMemory[tx]; // copy shared values back
}

__global__ void scanKernel(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here
  
  __shared__ float sharedMemory[2*BLOCK_SIZE];

  int bx = blockIdx.x; int tx = threadIdx.x;
  int x_in = bx * blockDim.x * 2 + tx;
  
  // copy the first block to shared memory
  if (x_in < len) sharedMemory[tx] = input[x_in];
  else sharedMemory[tx] = 0;
  
  // copy the second block to shared memory
  if (x_in + BLOCK_SIZE < len) sharedMemory[tx + BLOCK_SIZE] = input[x_in + BLOCK_SIZE];
  else sharedMemory[tx] = 0;
  
  __syncthreads();
  
  // perform the reduction step of Brent-Kung algorithm
  for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
  {
    int index = (tx+1) * stride * 2 - 1; 
    if(index < 2 * BLOCK_SIZE) sharedMemory[index] += sharedMemory[index-stride];
    
    __syncthreads();
  }
  
  // perform the inverse step of Brent-Kung algorithm
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2)
  {
    int index = (tx+1) * stride * 2 - 1;
    if(index + stride < 2 * BLOCK_SIZE) sharedMemory[index + stride] += sharedMemory[index]; 
    
    __syncthreads();
  }
  
  // save first and second blocks from shared memory to global
  if (x_in < len) output[x_in] = sharedMemory[tx];
  if (x_in + BLOCK_SIZE < len) output[x_in+BLOCK_SIZE] = sharedMemory[tx+BLOCK_SIZE];

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
  
  // perform Brent Kung addition
  scanKernel<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements);
  
  //perform hierarchical addition using cuda core (every thread will add the sum of the previous block)
  dimBlock.x = 2*BLOCK_SIZE; // change the block size - we need thread for every element
  vecAddKernel<<<dimGrid, dimBlock>>>(deviceOutput, deviceInput, numElements); 
  // I swap pointers since deviceOutput is the new input, and deviceInput is no longer needed
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceInput, numElements * sizeof(float), cudaMemcpyDeviceToHost));
  // copy deviceInput since it is the new output back to the host 
  
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
