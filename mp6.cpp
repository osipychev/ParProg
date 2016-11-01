// Histogram Equalization

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

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void float2charKernel(float *inputImagePtr, unsigned char *outputImagePtr){
  
  //int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int index = blockDim.x * ty * 3 + tx;

  outputImagePtr[index] = (unsigned char) (255 * inputImagePtr[index]);
  outputImagePtr[index + 1] = (unsigned char) (255 * inputImagePtr[index + 1]);
  outputImagePtr[index + 2] = (unsigned char) (255 * inputImagePtr[index + 2]);
}

__global__ void rgb2greyKernel(unsigned char *inputImagePtr, unsigned char *outputImagePtr){
  
  //int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int index = blockDim.x * ty * 3 + tx;

  unsigned char r = inputImagePtr[index];
  unsigned char g = inputImagePtr[index + 1];
  unsigned char b = inputImagePtr[index + 2];
  outputImagePtr[blockDim.x * ty + tx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  printf("index: %i, value: %u \n", index, outputImagePtr[blockDim.x * ty + tx]);
  
}

__global__ void histKernel(unsigned char *inputImagePtr){
  
  __shared__ int histogramShared[HISTOGRAM_LENGTH];
  
  int tx = threadIdx.x; int ty = threadIdx.y;
  int index = blockDim.x * ty + tx;
  
  histogramShared[tx] = 0;
  __syncthreads();
  
  histogramShared[inputImagePtr[index]]++;
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImage;
  unsigned char *deviceColorImage;
  unsigned char *deviceGreyImage;
  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  
  // allocate memory
  wbCheck(cudaMalloc((void **) &deviceInputImage, sizeof(wbImage_t))); // allocate the value in the gpu memory and print an error code
  wbCheck(cudaMalloc((void **) &deviceColorImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char))); // allocate the value in the gpu memory and print an error code
  wbCheck(cudaMalloc((void **) &deviceGreyImage, imageWidth * imageHeight * sizeof(unsigned char))); // allocate the value in the gpu memory and print an error code
  
  //copy memory to device
  wbCheck(cudaMemcpy(deviceInputImage, inputImage, sizeof(wbImage_t), cudaMemcpyHostToDevice)); // copy the variables into gpu memory
   
  dim3 dimGrid(1,1,1);
  dim3 dimBlock(imageWidth,imageHeight,1);

  float2charKernel<<<dimGrid, dimBlock>>>(deviceInputImage, deviceColorImage);
  rgb2greyKernel<<<dimGrid, dimBlock>>>(deviceColorImage, deviceGreyImage);
  
  cudaDeviceSynchronize();
  
  unsigned char *outputImageHost;
  outputImageHost = (unsigned char *)malloc(imageWidth * imageHeight * sizeof(unsigned char));
  wbCheck(cudaMemcpy(outputImageHost, deviceGreyImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost)); // copy the variables into gpu memory
   
  //for (int i=0; i<imageWidth * imageHeight; i++) printf("index %i, value %u", i, outputImageHost[i]);
  
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImage);
  cudaFree(deviceColorImage);
  cudaFree(deviceGreyImage);

  return 0;
}
