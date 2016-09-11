// MP 1
#include <wb.h>
//#include <math.h> 

__global__ void vecAddKernel(float *in1_d, float *in2_d, float *out_d, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x; // convert thread id into vector index
  if (i<len) out_d[i] = in1_d[i]+in2_d[i]; // do addition for threads which are in vector
}

// this is the sequential code just to compare the timing
__host__ void vecAddSeq(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  for (int i = 0; i < len; i++){
		out[i] = in1[i] + in2[i];
  }
}

__host__ int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");


  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int sizeBytes; // vector size in bytes
  int errCode; // CUDA error code
  sizeBytes = inputLength * sizeof(float);
  //wbLog(TRACE, "The input size in bytes is ", sizeBytes); // print the size of the vector if needed
  //wbLog(TRACE, "The input 1 is ", *(hostInput1+2)); // print the values from the vector if needed
  //wbLog(TRACE, "The input 2 is ", *(hostInput2+2));
  errCode = cudaMalloc((void **) &deviceInput1, sizeBytes); // allocate the value in the gpu memory and print an error code
  if (errCode) wbLog(TRACE, "Allocating GPU memory 1 is done with an error:", errCode); 
  errCode = cudaMalloc((void **) &deviceInput2, sizeBytes);
  if (errCode) wbLog(TRACE, "Allocating GPU memory 2 is done with an error:", errCode);
  errCode = cudaMalloc((void **) &deviceOutput, sizeBytes);
  if (errCode) wbLog(TRACE, "Allocating GPU memory 3 is done with an error:", errCode);
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  errCode = cudaMemcpy(deviceInput1, hostInput1, sizeBytes, cudaMemcpyHostToDevice); // copy the variables into gpu memory
  if (errCode) wbLog(TRACE, "Copying input memory 1 is done with an error:", errCode);
  errCode = cudaMemcpy(deviceInput2, hostInput2, sizeBytes, cudaMemcpyHostToDevice);
  if (errCode) wbLog(TRACE, "Copying input memory 2 is done with an error:", errCode);
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(inputLength/256,1,1);
  if (inputLength%256) DimGrid.x++;
  dim3 DimBlock(256,1,1);
  
  /* Do Sequential addition to compare timing  
  wbTime_start(Compute, "Performing sequential computation");
  vecAddSeq(hostInput1, hostInput2, hostOutput, inputLength);
  //free(hostOutput);
  wbTime_stop(Compute, "Performing sequential computation");
  */
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  vecAddKernel<<<DimGrid,DimBlock>>>(deviceInput1,deviceInput2,deviceOutput, inputLength);
  if (cudaGetLastError) wbLog(TRACE, "Performing CUDA computation is done with an error:", cudaGetLastError);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");
 
  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  errCode = cudaMemcpy(hostOutput, deviceOutput, sizeBytes, cudaMemcpyDeviceToHost);
  if (errCode) wbLog(TRACE, "Copying output memory to the CPU is done with an error:", errCode);  
  // wbLog(TRACE, "The result is ", *(hostOutput+2)); // print out the result if needed
                  
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
