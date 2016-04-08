#include "exp_cuda_sort_single.h"
#include "cuda_utils.h"
#include "timer.h"
#include "utils.h"
#include <algorithm>
#include <assert.h>
#include <chrono> // std::chrono::seconds
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <thread>

//#include <cuda.h>
#include <cuda_runtime.h>
void my_cuda_func(dim3 a, dim3 b, char *ab, int *bd);
void Runbitonic_sort_step(dim3 a, dim3 b, cudaStream_t stream, unsigned int *dev_values, int j,
                          int k);
void RunSortKernel(dim3 blocks, dim3 threads, cudaStream_t stream, int *theArray,
                   const unsigned int stage, const unsigned int passOfStage,
                   const unsigned int width);

#define DEFAULTPOWER 18
#define VERIFY 0
#define CUDATIME 1
CudaSortSingle::CudaSortSingle() : CudaExperiment(1, 1, "CudaSort", "Sorts Things") {}

CudaSortSingle::~CudaSortSingle() {}

static vector<cuda::CudaDevice> CtxDevices;
unsigned int CudaSortSingle::GetMinCu() {
  return 1;
}
unsigned int CudaSortSingle::GetMax() { return 1; }
void CudaSortSingle::Init(std::vector<cuda::CudaDevice> &devices) { CtxDevices = devices; }
void CudaSortSingle::Shutdown() {}

const int N = 16;
const int blocksize = 16;
const int threadsperblock = 512;
void CudaSortSingle::Start(unsigned int num_runs, const std::vector<int> options) {
  if (CtxDevices.size() < GetMinCu() || CtxDevices.size() > GetMax()) {
    std::cout << "\n invalid number of devices\n";
    return;
  }
  std::cout << "\n cuda Sort\n";
  // const int GPU_N = cuda::total_num_devices;
  // decode options
  uint16_t power;
  if (options.size() > 0) {
    power = options[0];
  } else {
    std::cout << "Power of numbers to sort?: (0 for default)" << std::endl;
    power = promptValidated<int, int>("Power: ", [](int i) { return (i >= 0 && i <= 256); });
  }
  if (power == 0) {
    power = DEFAULTPOWER;
  }

  uint32_t maxN = 1 << power;
  uint32_t maxNPC = (uint32_t)floor(maxN);
  size_t sz = maxN * sizeof(uint32_t);
  size_t szPC = maxNPC * sizeof(uint32_t);
  uint32_t *rndData = new uint32_t[maxN];
  for (cl_uint i = 0; i < maxN; i++) {
    uint32_t x = (uint32_t)0;
    rndData[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
    rndData[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
  }

  int temp;
  uint32_t numStages = 0;
  for (temp = maxNPC; temp > 2; temp >>= 1) {
    ++numStages;
  }
  size_t nThreads[1];
  nThreads[0] = maxNPC / (2 * 4);

  unsigned int runs = 0;
  running = true;
  should_run = true;

  // malloc
  int32_t *gpu1SwapBuffer;
  cudaStream_t stream;
  uint32_t *inBuffer;
  uint32_t *hostBuffer;

  checkCudaErrors(cudaSetDevice(CtxDevices[0].id));
  checkCudaErrors(cudaStreamCreate(&stream));
  // Allocate memory
  checkCudaErrors(cudaMalloc((void **)&inBuffer, szPC));
  checkCudaErrors(cudaMallocHost((void **)&hostBuffer, szPC));

  ResultFile r;
  r.name = "GpuCudaSort" + to_string(maxN);
  r.headdings = { "time_writebuffer", "time_sort", "time_read" };

  cudaEvent_t event1;
  cudaEvent_t event2;
  checkCudaErrors(cudaEventCreate(&event1));
  checkCudaErrors(cudaEventCreate(&event2));

  while (ShouldRun() && runs < num_runs) {
    vector<unsigned long long> times;
    unsigned int percentDone = (unsigned int)(floor(((float)runs / (float)num_runs) * 100.0f));
    std::cout << "\r" << Spinner(runs) << "\t" << runs << "\tPercent Done: " << percentDone << "%"
              << std::flush;
    // copy fresh rand data into host buffers
    std::copy(&rndData[0], &rndData[maxNPC], hostBuffer);

    // Copy data to GPU,
    float time_ms = 0;
    checkCudaErrors(cudaSetDevice(CtxDevices[0].id));
    checkCudaErrors(cudaEventRecord(event1, stream));
    checkCudaErrors(cudaMemcpyAsync(inBuffer, hostBuffer, szPC, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaEventRecord(event2, stream));
    // wait
    cudaStreamSynchronize(stream);

    checkCudaErrors(cudaEventElapsedTime(&time_ms, event1, event2));
    times.push_back(time_ms * 1000000.0f);

    unsigned int threadsperBlock = cuda::getBlockCount(threadsperblock, maxNPC);

    dim3 blocks((maxNPC / threadsperBlock), 1);
    dim3 threads(threadsperBlock, 1);

    checkCudaErrors(cudaEventRecord(event1, stream));
    int j, k;
    /* Major step */
    for (k = 2; k <= maxNPC; k <<= 1) {
      /* Minor step */
      for (j = k >> 1; j > 0; j = j >> 1) {
        Runbitonic_sort_step(blocks, threads, stream, inBuffer, j, k);
        getLastCudaError("reduceKernel() execution failed.\n");
      }
    }
    checkCudaErrors(cudaEventRecord(event2, stream));
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);

    // sort time
    checkCudaErrors(cudaEventElapsedTime(&time_ms, event1, event2));
    times.push_back(time_ms * 1000000.0f);

    // read back all data

    checkCudaErrors(cudaEventRecord(event1, stream));
    checkCudaErrors(cudaMemcpyAsync(hostBuffer, inBuffer, szPC, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaEventRecord(event2, stream));

    cudaStreamSynchronize(stream);
    checkCudaErrors(cudaEventElapsedTime(&time_ms, event1, event2));
    times.push_back(time_ms * 1000000.0f);

#if VERIFY == 1
    uint32_t *outData = new uint32_t[maxN];
    std::copy(&hostBuffers[0][0], &hostBuffers[0][maxNPC], outData);
    std::copy(&hostBuffers[1][0], &hostBuffers[1][maxNPC], &outData[maxNPC]);
    assert(CheckArrayOrder(outData, maxN, true));
    delete outData;
#endif

  

    r.times.push_back(times);
    ++runs;
  }

  checkCudaErrors(cudaEventDestroy(event1));
  checkCudaErrors(cudaEventDestroy(event2));
  checkCudaErrors(cudaFree(inBuffer));
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(cudaFreeHost(hostBuffer));
  checkCudaErrors(cudaSetDevice(CtxDevices[0].id));

  // Cleanup and shutdown
  cudaDeviceReset();

  delete[] rndData;
  r.CalcAvg();
  r.PrintToCSV(r.name);
  cout << "\n Sort finished\n";
  running = false;
}