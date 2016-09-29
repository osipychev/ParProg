#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
void convolution(float *in, float *out, const int z_size,
                       const int y_size, const int x_size, float *kernel){
for (int z_out = 0; z_out < z_size; z_out++)
  for (int y_out = 0; y_out < y_size; y_out++)
    for (int x_out = 0; x_out < x_size; x_out++) {
      float res = 0;
      for (int z_kernel = 0; z_kernel < 3; z_kernel++)
        for (int y_kernel = 0; y_kernel < 3; y_kernel++)
          for (int x_kernel = 0; x_kernel < 3; x_kernel++) {
            int z_in = z_out - 1 + z_kernel;
            int y_in = y_out - 1 + y_kernel;
            int x_in = x_out - 1 + x_kernel;
            // Pad boundary with 0
            if (z_in >= 0 && z_in < z_size &&
                y_in >= 0 && y_in < y_size &&
                x_in >= 0 && x_in < x_size) {
              res += kernel[z_kernel * 9 + y_kernel * 3 + x_kernel] * in[y_size*x_size*z_in + x_size*y_in + x_in];
            }
          }

      out[y_size*x_size*z_out + x_size*y_out + x_out] = res;
    }
}

//@@ Define constant memory for device kernel here

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;
  float *deviceKernel;
    
  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbCheck(cudaMalloc((void **) &deviceInput, z_size * y_size * x_size * sizeof(float))); // allocate the value in the gpu memory and print an error code
  wbCheck(cudaMalloc((void **) &deviceOutput, z_size * y_size * x_size * sizeof(float))); // allocate the value in the gpu memory and print an error code
  
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbCheck(cudaMemcpy(deviceInput, hostInput+3, z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice)); // copy the variables into gpu memory
  //wbCheck(cudaMemcpy(deviceKernel, hostKernel, kernelLength, cudaMemcpyHostToDevice)); // copy the variables into gpu memory
  
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here

  //@@ Launch the GPU kernel here
  convolution(hostInput+3, hostOutput+3, z_size, y_size, x_size, hostKernel);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
