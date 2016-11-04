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
  if (indX < width && indY < height){
    int index = (width * indY + indX) * channels;

  outputImagePtr[index] = (unsigned char) (255 * inputImagePtr[index]);
  outputImagePtr[index + 1] = (unsigned char) (255 * inputImagePtr[index + 1]);
  outputImagePtr[index + 2] = (unsigned char) (255 * inputImagePtr[index + 2]);
  }
    //if (index == 0) printf("pixel %i, r: %d, g: %d, b: %d \n",index, outputImagePtr[index],outputImagePtr[index+1],outputImagePtr[index+2]);
}


__global__ void rgb2greyKernel(unsigned char *inputImagePtr, unsigned char *outputImagePtr, int width, int height, int channels){
  
  int indX = blockIdx.x * blockDim.x + threadIdx.x; 
  int indY = blockIdx.y * blockDim.y + threadIdx.y;
  int index = (width * indY + indX) * channels;

  unsigned char r = inputImagePtr[index];
  unsigned char g = inputImagePtr[index + 1];
  unsigned char b = inputImagePtr[index + 2];
  
  if (indX < width && indY < height){
    outputImagePtr[width * indY + indX] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    //printf("%i, %i, %d \n", indX, indY, outputImagePtr[width * indY + indX]);
  }
  if (width * indY + indX == 0) printf("pixel %i, grey: %d \n",width * indY + indX, outputImagePtr[width * indY + indX]);
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
  if (index<256) printf("hist: %i, %f \n", index, histPtr[index]);
}


__global__ void histogram_privatized_kernel(unsigned char* input, unsigned int* bins,unsigned int num_elements, unsigned int num_bins) {
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  extern __shared__ unsigned int histo_s[];
  for(unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx +=blockDim.x) {
    histo_s[binIdx] = 0u;
  }
  __syncthreads();
  
  for(unsigned int i = tid; i < num_elements; i += blockDim.x*gridDim.x) {
    int alphabet_position = buffer[i] –“a”;
    if (alphabet_position >= 0 && alpha_position < 26) atomicAdd(&(histo_s[alphabet_position/4]), 1);}
  __syncthreads();
  
  for(unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {                                                                      
    atomicAdd(&(histo[binIdx]), histo_s[binIdx]);
  }
}


void histSerial(unsigned char *inputImagePtr, float *histPtr,  int width, int height){

  for (int i = 0; i < 256; i++) histPtr[i] = 0;
    
  for (int i = 0; i < width*height; i++){
    histPtr[inputImagePtr[i]] += 1;
  }
  for (int i = 0; i < 256; i++) {
    histPtr[i] /= (float)(width*height);
    //printf("hist: %i, %f \n", i, histPtr[i]);
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
  //printf("cdf: %i, %f \n",tx, sharedMemory[tx]);
  //printf("cdf: %i, %f \n", tx+blockDim.x, sharedMemory[tx+blockDim.x]);
}


__global__ void histCorKernel(unsigned char *imagePtr, float *histCdfPtr, int width, int height, int channels){
  
  int indX = blockIdx.x * blockDim.x + threadIdx.x; 
  int indY = blockIdx.y * blockDim.y + threadIdx.y;
  if (indX < width && indY < height){
  int index = (width * indY + indX) * channels;

  for (int i=0; i<3; i++){
    if (index == 0)
      printf("i: %d, pix: %d, cdf: %f, value: %f, result: %d \n",index+i,imagePtr[index+i],histCdfPtr[imagePtr[index+i]],
             255.0*histCdfPtr[imagePtr[index+i]],
            (unsigned char)(min(max(255.0*(histCdfPtr[imagePtr[index+i]] - histCdfPtr[0])/(1 - histCdfPtr[0]), 0.0), 255.0)));
    
    imagePtr[index+i] = (unsigned char)(min(max(255.0*(histCdfPtr[imagePtr[index+i]] - histCdfPtr[0])/(1 - histCdfPtr[0]), 0.0), 255.0));
  }
  }
}


__global__ void char2floatKernel(unsigned char *inputImagePtr, float *outputImagePtr, int width, int height, int channels){
  
  int indX = blockIdx.x * blockDim.x + threadIdx.x; 
  int indY = blockIdx.y * blockDim.y + threadIdx.y;
  if (indX < width && indY < height){
  int index = (width * indY + indX) * channels;

  outputImagePtr[index] = (float)(inputImagePtr[index])/255.0;
  outputImagePtr[index + 1] = (float)(inputImagePtr[index + 1])/255.0;
  outputImagePtr[index + 2] = (float)(inputImagePtr[index + 2])/255.0;
  if (index == 0) printf("%f, %f, %f \n",outputImagePtr[index],outputImagePtr[index+1],outputImagePtr[index+2]);
  }
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
  
  histCorKernel<<<dimGrid, dimBlock>>>(deviceColorImage, deviceHistogramCDF, imageWidth, imageHeight, imageChannels);
  char2floatKernel<<<dimGrid, dimBlock>>>(deviceColorImage, deviceInputImage, imageWidth, imageHeight, imageChannels);
  
  cudaDeviceSynchronize();
  wbTime_stop(Generic, "All kernel runs");

  wbCheck(cudaMemcpy(hostInputImageData, deviceInputImage, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost)); // copy the variables into gpu memory
  
  //unsigned char *histHost;
  //histHost = (unsigned char *)malloc(HISTOGRAM_LENGTH * sizeof(float));
  //wbCheck(cudaMemcpy(histHost, deviceHistogram, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost)); // copy the variables into gpu memory
  //for (int i=0; i<HISTOGRAM_LENGTH;i++)
    //wbLog(TRACE, "Element is ", (int)histHost[0]);
  
  //dim3 dimGrid2(HISTOGRAM_LENGTH/(2*BLOCK_SIZE), 1, 1);
  //if (HISTOGRAM_LENGTH%(2*BLOCK_SIZE)) dimGrid2.x++;
  //dim3 dimBlock2(BLOCK_SIZE, 1, 1);
  
  wbImage_setData(outputImage,hostInputImageData);
  
   
  //for (int i=0; i<imageWidth * imageHeight; i++) printf("index %i, value %u", i, outputImageHost[i]);
  
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImage);
  cudaFree(deviceColorImage);
  cudaFree(deviceGreyImage);

  return 0;
}
