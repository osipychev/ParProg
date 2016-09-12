#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

/*/ Compute C = A * B
void matrixMultiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
  for (int i = 0; i < numARows; ++i){
        for (int j = 0; j < numBColumns; ++j) {
            float sum = 0;
            for (int k = 0; k < numAColumns; ++k) {
                float a = A[i * numAColumns + k];
                float b = B[k * numBColumns + j];
                sum += a * b;
            }
            C[i * numCColumns + j] = sum;
        }}
}*/

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numAColumns, int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if ((Row < numCRows) && (Col < numCColumns)) {
    float sum = 0;
    for (int k = 0; k < numAColumns; ++k)
      sum += A[Row*numAColumns+k] * B[k*numCColumns+Col];
    
    C[Row*numCColumns+Col] = sum;
  } 
}

int main(int argc, char ** argv) {
  wbArg_t args;
  float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float)); // allocate memory as for a 1 dim array

  wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
  wbCheck(cudaMalloc((void **) &deviceA, numARows * numAColumns * sizeof(float))); // allocate the value in the gpu memory and print an error code
  wbCheck(cudaMalloc((void **) &deviceB, numBRows * numBColumns * sizeof(float)));
  wbCheck(cudaMalloc((void **) &deviceC, numCRows * numCColumns * sizeof(float)));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice)); // copy the variables into gpu memory
  wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(numCColumns/32,numCRows/32,1);
  if (numCColumns%32) DimGrid.x++;
  if (numCRows%32) DimGrid.y++;
  dim3 DimBlock(32,32,1);
   
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
  matrixMultiply<<<DimGrid,DimBlock>>>(deviceA, deviceB, deviceC, numAColumns, numCRows, numCColumns);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost)); // copy the variables into gpu memory
  
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
