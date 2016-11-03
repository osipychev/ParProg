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
#define BLOCK_SIZE 32

//@@ insert code here
__global__ void float2charKernel(float *inputImagePtr, unsigned char *outputImagePtr, int width, int height, int channels){
  
  int indX = blockIdx.x * blockDim.x + threadIdx.x; 
  int indY = blockIdx.y * blockDim.y + threadIdx.y;
  int index = width * channels * indY + indX;

  outputImagePtr[index] = (unsigned char) (255 * inputImagePtr[index]);
  outputImagePtr[index + 1] = (unsigned char) (255 * inputImagePtr[index + 1]);
  outputImagePtr[index + 2] = (unsigned char) (255 * inputImagePtr[index + 2]);
  //if (index == 0) printf("%u \n",outputImagePtr[index]);
}


__global__ void rgb2greyKernel(unsigned char *inputImagePtr, unsigned char *outputImagePtr, int width, int height, int channels){
  
  int indX = blockIdx.x * blockDim.x + threadIdx.x; 
  int indY = blockIdx.y * blockDim.y + threadIdx.y;
  int index = width * channels * indY + indX;

  unsigned char r = inputImagePtr[index];
  unsigned char g = inputImagePtr[index + 1];
  unsigned char b = inputImagePtr[index + 2];
  
  if (indX < width && indY < height){
    outputImagePtr[width * indY + indX] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    //printf("%i, %i, %d \n", indX, indY, outputImagePtr[width * indY + indX]);
  }
}


__global__ void histKernel(unsigned char *inputImagePtr, float *histPtr,  int width, int height){
  
  __shared__ float histogramShared[HISTOGRAM_LENGTH];
  
  int indX = blockIdx.x * blockDim.x + threadIdx.x; 
  int indY = blockIdx.y * blockDim.y + threadIdx.y;
  int index = width * indY + indX;
  
  if (index < HISTOGRAM_LENGTH) {
    histogramShared[indX] = 0.0;
    histPtr[indX] = 0.0;
  }
  __syncthreads();
  
  int stride = blockDim.x * gridDim.x;    
  while (indX < width*hight) {
    atomicAdd( &(histogramShared[inputImagePtr[i]]), 1);
    indX += stride;
  }
                              
  //atomicAdd(&histogramShared[inputImagePtr[index]],1);
  //if (index<16) printf("ind: %i, val: %d, hist: %d", index, inputImagePtr[index],histPtr[inputImagePtr[index]]);
  //atomicAdd(&histPtr[(int)inputImagePtr[index]],1);
  __syncthreads();
  //if (index<16) printf("ind: %i, val: %d, hist: %d \n", index, inputImagePtr[index],histPtr[inputImagePtr[index]]);
  
  
  
  if (indX < HISTOGRAM_LENGTH) {
    atomicAdd(&histPtr[indX],histogramShared[index]);
  }
  
  __syncthreads();
  //if (index < HISTOGRAM_LENGTH) printf(" %f, ", histPtr[index]);
}

__global__ void scanKernel(float *input, float *output, int len, int numOfPix) {
    
  __shared__ float sharedMemory[2*BLOCK_SIZE];

  int bx = blockIdx.x; int tx = threadIdx.x;
  int x_in = bx * blockDim.x * 2 + tx;
  
  // copy the first block to shared memory
  if (x_in < len) sharedMemory[tx] = input[x_in];
  else sharedMemory[tx] = 0;
  
  // copy the second block to shared memory
  if (x_in + BLOCK_SIZE < len) sharedMemory[tx + BLOCK_SIZE] = input[x_in + BLOCK_SIZE];
  else sharedMemory[tx + BLOCK_SIZE] = 0;
  
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
  if (x_in < len) output[x_in] = sharedMemory[tx]/(float)numOfPix;
  if (x_in + BLOCK_SIZE < len) output[x_in+BLOCK_SIZE] = sharedMemory[tx+BLOCK_SIZE]/(float)numOfPix;

}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  //float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImage;
  unsigned char *deviceColorImage;
  unsigned char *deviceGreyImage;
  float *deviceHistogram;
  float *deviceHistogramCDF;
  
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
  hostInputImageData = wbImage_getData(inputImage);
  
  // allocate memory
  wbCheck(cudaMalloc((void **) &deviceInputImage, imageWidth * imageHeight * imageChannels * sizeof(float))); // allocate the value in the gpu memory and print an error code
  wbCheck(cudaMalloc((void **) &deviceColorImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char))); // allocate the value in the gpu memory and print an error code
  wbCheck(cudaMalloc((void **) &deviceGreyImage, imageWidth * imageHeight * sizeof(unsigned char))); // allocate the value in the gpu memory and print an error code
  wbCheck(cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(float))); // allocate the value in the gpu memory and print an error code
  //wbCheck(cudaMalloc((void **) &deviceHistogramCDF, HISTOGRAM_LENGTH * sizeof(float))); // allocate the value in the gpu memory and print an error code
  
  //copy memory to device
  wbCheck(cudaMemcpy(deviceInputImage, hostInputImageData,imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice)); // copy the variables into gpu memory
   
  dim3 dimGrid(imageWidth/BLOCK_SIZE,imageHeight/BLOCK_SIZE,1);
  if (imageWidth%BLOCK_SIZE) dimGrid.x++;
  if (imageHeight%BLOCK_SIZE) dimGrid.y++;
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
  
  wbTime_start(Generic, "All kernel runs");
  
  float2charKernel<<<dimGrid, dimBlock>>>(deviceInputImage, deviceColorImage, imageWidth, imageHeight, imageChannels);
  rgb2greyKernel<<<dimGrid, dimBlock>>>(deviceColorImage, deviceGreyImage, imageWidth, imageHeight, imageChannels);
  histKernel<<<dimGrid, dimBlock>>>(deviceGreyImage, deviceHistogram, imageWidth, imageHeight);
  wbTime_stop(Generic, "All kernel runs");

  //unsigned char *histHost;
  //histHost = (unsigned char *)malloc(HISTOGRAM_LENGTH * sizeof(float));
  //wbCheck(cudaMemcpy(histHost, deviceHistogram, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost)); // copy the variables into gpu memory
  //for (int i=0; i<HISTOGRAM_LENGTH;i++)
    //wbLog(TRACE, "Element is ", (int)histHost[0]);
  
  //dim3 dimGrid2(HISTOGRAM_LENGTH/(2*BLOCK_SIZE), 1, 1);
  //if (HISTOGRAM_LENGTH%(2*BLOCK_SIZE)) dimGrid2.x++;
  //dim3 dimBlock2(BLOCK_SIZE, 1, 1);
  
  //scanKernel<<<dimGrid2, dimBlock2>>>(deviceHistogram, deviceHistogramCDF, HISTOGRAM_LENGTH, imageWidth*imageHeight);
  
  cudaDeviceSynchronize();
  
   
  //for (int i=0; i<imageWidth * imageHeight; i++) printf("index %i, value %u", i, outputImageHost[i]);
  
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImage);
  cudaFree(deviceColorImage);
  cudaFree(deviceGreyImage);

  return 0;
}
