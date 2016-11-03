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


__global__ void histKernel2D(unsigned char *inputImagePtr, float *histPtr,  int width, int height){
  
  __shared__ float histogramShared[HISTOGRAM_LENGTH];
  
  int indX = blockIdx.x * blockDim.x + threadIdx.x; 
  int indY = blockIdx.y * blockDim.y + threadIdx.y;
  int index = width * indY + indX;
   
  if (threadIdx.x * threadIdx.y < HISTOGRAM_LENGTH) {
    histogramShared[threadIdx.x * threadIdx.y] = 0.0;
    histPtr[threadIdx.x * threadIdx.y] = 0.0;
  }
  __syncthreads();
  
  atomicAdd( &(histogramShared[inputImagePtr[index]]), 1);
  __syncthreads();
  
  if (threadIdx.x * threadIdx.y < 256)
    atomicAdd( &(histPtr[threadIdx.x * threadIdx.y]), histogramShared[threadIdx.x * threadIdx.y]);
  
  //histPtr[index] = histPtr[index]/(float)(width*height);
   __syncthreads();
  if (index<256) printf("%f \n", histPtr[index]);
}


__global__ void histKernel(unsigned char *inputImagePtr, float *histPtr,  int width, int height){
  
  __shared__ float histogramShared[HISTOGRAM_LENGTH];
  
  if (threadIdx.x < 256) {
    histogramShared[threadIdx.x] = 0;
    histPtr[threadIdx.x] = 0;
  }
  __syncthreads();
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;    
  while (i < width*height) {
    atomicAdd( &(histogramShared[inputImagePtr[i]]), 1);
    i += stride;    
  }
  
  __syncthreads();
  if (threadIdx.x < 256)
    atomicAdd( &(histPtr[threadIdx.x]), histogramShared[threadIdx.x] );

  if (threadIdx.x<256 && blockIdx.x == 0 && blockIdx.y == 0) printf("%f \n", histPtr[threadIdx.x]);
}


void histSerial(unsigned char *inputImagePtr, float *histPtr,  int width, int height){

  for (int i = 0; i < 256; i++) histPtr[i] = 0;
    
  for (int i = 0; i < width*height; i++){
    histPtr[inputImagePtr[i]] += 1;
  }
  for (int i = 0; i < 256; i++) {
    histPtr[i] /= (float)(width*height);
    //printf("%f \n", histPtr[i]);
  }
}


__global__ void scanKernel(float *input, float *output, int len) {
    
  __shared__ float sharedMemory[HISTOGRAM_LENGTH];

  int bx = blockIdx.x; int tx = threadIdx.x;
  int x_in = bx * blockDim.x * 2 + tx;
  
  // copy the first block to shared memory
  if (x_in < len) sharedMemory[tx] = input[x_in];
  else sharedMemory[tx] = 0;
  
  // copy the second block to shared memory
  if (x_in + blockDim.x < len) sharedMemory[tx + blockDim.x] = input[x_in + blockDim.x];
  else sharedMemory[tx + blockDim.x] = 0;
  
  __syncthreads();
  
  // perform the reduction step of Brent-Kung algorithm
  for (int stride = 1; stride <= blockDim.x; stride *= 2)
  {
    int index = (tx+1) * stride * 2 - 1; 
    if(index < 2 * blockDim.x) sharedMemory[index] += sharedMemory[index-stride];
    
    __syncthreads();
  }
  
  // perform the inverse step of Brent-Kung algorithm
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    int index = (tx+1) * stride * 2 - 1;
    if(index + stride < 2 * blockDim.x) sharedMemory[index + stride] += sharedMemory[index]; 
    
    __syncthreads();
  }
  
  // save first and second blocks from shared memory to global
  if (x_in < len) output[x_in] = sharedMemory[tx];
  if (x_in + blockDim.x < len) output[x_in+blockDim.x] = sharedMemory[tx+blockDim.x];
  //printf("%f \n",sharedMemory[tx]);
  //printf("%f \n",sharedMemory[tx+blockDim.x]);
}



__global__ void histCorKernel(unsigned char *imagePtr, float *histPtr, int width, int height, int channels){
  
  int indX = blockIdx.x * blockDim.x + threadIdx.x; 
  int indY = blockIdx.y * blockDim.y + threadIdx.y;
  int index = width * channels * indY + indX;

  imagePtr[index] *= min(max(255*(cdf[val] - cdfmin)/(1 - cdfmin), 0), 255);
  imagePtr[index + 1] *=
  imagePtr[index + 2] *=
  //if (index == 0) printf("%u \n",outputImagePtr[index]);
}

for ii from 0 to (width * height * channels) do
    image[ii] = correct_color(ucharImage[ii])
end


__global__ void char2floatKernel(unsigned char *inputImagePtr, float *outputImagePtr, int width, int height, int channels){
  
  int indX = blockIdx.x * blockDim.x + threadIdx.x; 
  int indY = blockIdx.y * blockDim.y + threadIdx.y;
  int index = width * channels * indY + indX;

  outputImagePtr[index] = (float)inputImagePtr[index])/255.0;
  outputImagePtr[index + 1] = (float)inputImagePtr[index + 1])/255.0;
  outputImagePtr[index + 2] = (float)inputImagePtr[index + 2])/255.0;
  //if (index == 0) printf("%u \n",outputImagePtr[index]);
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
  
  // allocate memory in cuda device
  wbCheck(cudaMalloc((void **) &deviceInputImage, imageWidth * imageHeight * imageChannels * sizeof(float))); // allocate the value in the gpu memory and print an error code
  wbCheck(cudaMalloc((void **) &deviceColorImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char))); // allocate the value in the gpu memory and print an error code
  wbCheck(cudaMalloc((void **) &deviceGreyImage, imageWidth * imageHeight * sizeof(unsigned char))); // allocate the value in the gpu memory and print an error code
  wbCheck(cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(float))); // allocate the value in the gpu memory and print an error code
  wbCheck(cudaMalloc((void **) &deviceHistogramCDF, HISTOGRAM_LENGTH * sizeof(float))); // allocate the value in the gpu memory and print an error code
  
  //copy input image to cuda device
  wbCheck(cudaMemcpy(deviceInputImage, hostInputImageData,imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice)); // copy the variables into gpu memory
   
  dim3 dimGrid(imageWidth/BLOCK_SIZE,imageHeight/BLOCK_SIZE,1);
  if (imageWidth%BLOCK_SIZE) dimGrid.x++;
  if (imageHeight%BLOCK_SIZE) dimGrid.y++;
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
  
  wbTime_start(Generic, "All kernel runs");
  //float to char cuda
  float2charKernel<<<dimGrid, dimBlock>>>(deviceInputImage, deviceColorImage, imageWidth, imageHeight, imageChannels);
  // grb to grey cuda
  rgb2greyKernel<<<dimGrid, dimBlock>>>(deviceColorImage, deviceGreyImage, imageWidth, imageHeight, imageChannels);
  
  // histogram serial
  float histHost[256];
  unsigned char image[imageWidth*imageHeight];
  cudaMemcpy(image, deviceGreyImage, imageWidth*imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  histSerial(image, histHost, imageWidth, imageHeight);
  //histKernel<<<imageHeight*imageWidth/256, 256>>>(deviceGreyImage, deviceHistogram, imageWidth, imageHeight);
  
  // scan histogram cuda
  wbCheck(cudaMemcpy(deviceHistogram, histHost, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice)); // copy the variables into gpu memory
  scanKernel<<<1, HISTOGRAM_LENGTH/2>>>(deviceHistogram, deviceHistogramCDF, HISTOGRAM_LENGTH);
  
  
  
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
