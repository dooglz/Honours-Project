#include "cudaSort.h"
#include "utils.h"
#include "Timer.h"
#include <chrono> // std::chrono::seconds
#include <thread>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <assert.h>

//#include <cuda.h>
#include <cuda_runtime.h>

void my_cuda_func(dim3 a, dim3 b, char *ab, int *bd);

#define DEFAULTPOWER 12
#define VERIFY true
CudaSort::CudaSort() : Experiment(1, 4, "CudaSort", "Sorts Things") {}

CudaSort::~CudaSort() {}

static cl_context ctx;
static vector<cl::CLDevice> CtxDevices;
static std::vector<cl_command_queue> cq;
unsigned int CudaSort::GetMinCu() { return 1; }
unsigned int CudaSort::GetMax() { return 4; }
void CudaSort::Init(cl_context &context, std::vector<cl_command_queue> &commandQ,
                    std::vector<cl::CLDevice> &devices, cl::Platform platform) {
  CtxDevices = devices;
  ctx = context;
  cq = commandQ;
}
void CudaSort::Shutdown() {}

const int N = 16;
const int blocksize = 16;

void CudaSort::Start(unsigned int num_runs, const std::vector<int> options) {
  cout << "\n cuda Sort\n";

  char a[N] = "Hello \0\0\0\0\0\0";
  int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  char *ad;
  int *bd;
  const int csize = N * sizeof(char);
  const int isize = N * sizeof(int);

  printf("%s", a);

  cudaMalloc((void **)&ad, csize);
  cudaMalloc((void **)&bd, isize);
  cudaMemcpy(ad, a, csize, cudaMemcpyHostToDevice);
  cudaMemcpy(bd, b, isize, cudaMemcpyHostToDevice);

  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);

  my_cuda_func(dimGrid, dimBlock, ad, bd);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(a, ad, csize, cudaMemcpyDeviceToHost);
  cudaFree(ad);
  cudaFree(bd);

  cout << "\n Sort finished\n";
  running = false;
};
