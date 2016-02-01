#include "exp_cuda_sort.h"
#include "cuda_utils.h"
#include "utils.h"
#include "Timer.h"
#include <chrono> // std::chrono::seconds
#include <thread>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <string>

//#include <cuda.h>
#include <cuda_runtime.h>
void my_cuda_func(dim3 a, dim3 b, char *ab, int *bd);
void Runbitonic_sort_step(dim3 a, dim3 b, cudaStream_t stream, unsigned int *dev_values, int j,
                          int k);
void RunSortKernel(dim3 blocks, dim3 threads, cudaStream_t stream, int *theArray,
                   const unsigned int stage, const unsigned int passOfStage,
                   const unsigned int width);

#define DEFAULTPOWER 18
#define VERIFY 1
CudaSort::CudaSort() : CudaExperiment(1, 4, "CudaSort", "Sorts Things") {}

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
const int threadsperblock = 512;
uint32_t mxount = 0;
void CudaSort::Start(unsigned int num_runs, const std::vector<int> options) {
  std::cout << "\n cuda Sort\n";
  // const int GPU_N = cuda::total_num_devices;
  // decode options
  uint16_t power;
  if (options.size() > 0) {
    power = options[0];
  } else {
    cout << "Power of numbers to sort?: (0 for default)" << std::endl;
    power = promptValidated<int, int>("Power: ", [](int i) { return (i >= 0 && i <= 256); });
  }
  if (power == 0) {
    power = DEFAULTPOWER;
  }
  int optmode = 0;
  if (options.size() > 1) {
    optmode = options[1];
  } else {
    cout << "dAta transfer mode (0 dumb, 1 peer, 2 UVA)" << std::endl;
    optmode = promptValidated<int, int>("optmode: ", [](int i) { return (i >= 0 && i <= 2); });
  }

  const int GPU_N = 2;
  uint32_t maxN = 1 << DEFAULTPOWER;
  uint32_t maxNPC = (uint32_t)floor(maxN / GPU_N);
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
  if (optmode == 1 || optmode == 2) {
   // checkCudaErrors(cudaSetDevice(CtxDevices[0].id));
    checkCudaErrors(cudaMalloc(&gpu1SwapBuffer, (maxNPC / 2) * sizeof(uint32_t)));
    if (optmode == 2) {
      std::cout << "attemptiung UVA p2p" << endl;
      if (GPU_N != 2) {
        cerr << "Need 2 gpus!" << endl;
        return;
      }
      if (!cuda::enableUVA(0, 1)) {
        return;
      }
    }
  }
  cudaStream_t *streams = new cudaStream_t[GPU_N];
  uint32_t **inBuffers = new uint32_t *[GPU_N];
  uint32_t **hostBuffers = new uint32_t *[GPU_N];
  for (auto i = 0; i < GPU_N; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaStreamCreate(&streams[i]));
    // Allocate memory
    checkCudaErrors(cudaMalloc((void **)&inBuffers[i], szPC));
    checkCudaErrors(cudaMallocHost((void **)&hostBuffers[i], szPC));
  }

  ResultFile r;
  r.name = "GpuParallelCudaSort" + to_string(maxN) + "_" + to_string(optmode);
  r.headdings = {"time_writebuffer"};
  r.attributes.push_back("Optimised Swapping mode, " + optmode);

  while (ShouldRun() && runs < num_runs) {
    vector<unsigned long long> times;
    unsigned int percentDone = (unsigned int)(floor(((float)runs / (float)num_runs) * 100.0f));
    cout << "\r" << Spinner(runs) << "\t" << runs << "\tPercent Done: " << percentDone << "%"
         << std::flush;

    // copy fresh rand data into host buffers
    for (auto i = 0; i < GPU_N; i++) {
      const uint32_t offset = (i * maxNPC);
      std::copy(&rndData[offset], &rndData[offset + maxNPC], hostBuffers[i]);
    }
    Timer time_writebuffer;
    // Copy data to GPU,
    for (auto i = 0; i < GPU_N; i++) {
      checkCudaErrors(cudaSetDevice(i));
      checkCudaErrors(
          cudaMemcpyAsync(inBuffers[i], hostBuffers[i], szPC, cudaMemcpyHostToDevice, streams[i]));
    }
    // wait
    for (auto i = 0; i < GPU_N; i++) {
      cudaStreamSynchronize(streams[i]);
    }

    time_writebuffer.Stop();
    times.push_back(time_writebuffer.Duration_NS());

    // run the sort.
    unsigned int swapcount = 0;
    for (cl_uint swapsize = maxNPC / 2; swapsize > 0; swapsize /= 2) {
      if (runs == 0) {
        r.headdings.push_back("Sort_" + to_string(swapcount));
        r.headdings.push_back("Swap_" + to_string(swapsize));
      }
      Timer time_sort_inner;
      unsigned int threadsperBlock = cuda::getBlockCount(threadsperblock, maxNPC);

      dim3 blocks((maxNPC / threadsperBlock), 1);
      dim3 threads(threadsperBlock, 1);

      for (size_t i = 0; i < GPU_N; i++) {
        checkCudaErrors(cudaSetDevice(i));
        int j, k;
        /* Major step */
        for (k = 2; k <= maxNPC; k <<= 1) {
          /* Minor step */
          for (j = k >> 1; j > 0; j = j >> 1) {
            Runbitonic_sort_step(blocks, threads, streams[i], inBuffers[i], j, k);
            getLastCudaError("reduceKernel() execution failed.\n");
          }
        }
      }
      for (auto i = 0; i < GPU_N; i++) {
        checkCudaErrors(cudaSetDevice(i));
        cudaDeviceSynchronize();
        cudaStreamSynchronize(streams[i]);
      }

      time_sort_inner.Stop();
      times.push_back(time_sort_inner.Duration_NS());
      Timer time_swap_inner;
      // Do swaps

      uint32_t a = swapsize * sizeof(uint32_t);
      if (optmode == 2) {
        // copy top of card 1 to it's swapbuffer
        checkCudaErrors(cudaSetDevice(1));
        checkCudaErrors(cudaMemcpy(gpu1SwapBuffer, inBuffers[1], a, cudaMemcpyDeviceToDevice));
        cudaDeviceSynchronize();

        // copy bottom of card 0 to top of card 1
        checkCudaErrors(
            cudaMemcpy(inBuffers[1], &inBuffers[0][(maxNPC - swapsize)], a, cudaMemcpyDefault));
        cudaDeviceSynchronize();

        // write the top of card 1 to the bottom of card 0
        checkCudaErrors(cudaSetDevice(0));
        checkCudaErrors(cudaMemcpy(&inBuffers[0][(maxNPC - swapsize)], gpu1SwapBuffer, a,
                                   cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();

      } else if (optmode == 1) {
        // copy top of card 1 to it's swapbuffer
        checkCudaErrors(cudaSetDevice(1));
        checkCudaErrors(cudaMemcpy(gpu1SwapBuffer, inBuffers[1], a, cudaMemcpyDeviceToDevice));
        cudaDeviceSynchronize();

        // copy bottom of card 0 to top of card 1
        checkCudaErrors(cudaMemcpyPeer(inBuffers[1], 1, &inBuffers[0][(maxNPC - swapsize)], 0, a));
        cudaDeviceSynchronize();

        // write the top of card 1 to the bottom of card 0
        checkCudaErrors(cudaSetDevice(0));
        checkCudaErrors(
            cudaMemcpyPeer(&inBuffers[0][(maxNPC - swapsize)], 0, gpu1SwapBuffer, 1, a));
        cudaDeviceSynchronize();
      } else {
        uint32_t *tmpData = new uint32_t[swapsize];
        // read back all data
        for (auto i = 0; i < GPU_N; i++) {
          checkCudaErrors(cudaSetDevice(i));
          checkCudaErrors(cudaMemcpyAsync(hostBuffers[i], inBuffers[i], szPC,
                                          cudaMemcpyDeviceToHost, streams[i]));
        }
        for (auto i = 0; i < GPU_N; i++) {
          cudaStreamSynchronize(streams[i]);
        }
        // read in the top of card 1 to temp
        std::copy(hostBuffers[1], &hostBuffers[1][swapsize], tmpData);

        // copy bottom of card 0 to top of card 1
        std::copy(&hostBuffers[0][(maxNPC - swapsize)],
                  &hostBuffers[0][(maxNPC - swapsize) + swapsize], hostBuffers[1]);

        // write the top of card 1 to the bottom of card 0
        std::copy(tmpData, &tmpData[swapsize], &hostBuffers[0][(maxNPC - swapsize)]);
        delete[] tmpData;

        // Copy data back to GPU
        for (auto i = 0; i < GPU_N; i++) {
          // Set device
          checkCudaErrors(cudaSetDevice(i));

          // Copy input data from CPU
          checkCudaErrors(cudaMemcpy(inBuffers[i], hostBuffers[i], szPC, cudaMemcpyHostToDevice));
          cudaDeviceSynchronize();
        }
      }

      // wait
      for (auto i = 0; i < GPU_N; i++) {
        checkCudaErrors(cudaSetDevice(i));
        cudaStreamSynchronize(streams[i]);
        cudaDeviceSynchronize();
      }

      time_swap_inner.Stop();
      times.push_back(time_swap_inner.Duration_NS());
      ++swapcount;
    }

    if (runs == 0) {
      r.headdings.push_back("Copy Back");
    }
    Timer time_copyback;
    // read back all data
    for (auto i = 0; i < GPU_N; i++) {
      checkCudaErrors(cudaSetDevice(i));
      checkCudaErrors(
          cudaMemcpyAsync(hostBuffers[i], inBuffers[i], szPC, cudaMemcpyDeviceToHost, streams[i]));
    }
    for (auto i = 0; i < GPU_N; i++) {
      cudaStreamSynchronize(streams[i]);
    }
    time_copyback.Stop();
    times.push_back(time_copyback.Duration_NS());

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

  for (auto i = 0; i < GPU_N; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaFree(inBuffers[i]));
    checkCudaErrors(cudaStreamDestroy(streams[i]));
    checkCudaErrors(cudaFreeHost(hostBuffers[i]));
  }

  delete streams;
  delete[] hostBuffers;
  delete[] inBuffers;

  // Cleanup and shutdown
  for (auto i = 0; i < GPU_N; i++) {
    cudaDeviceReset();
  }
  delete[] rndData;
  r.CalcAvg();
  r.PrintToCSV(r.name);
  cout << "\n Sort finished\n";
  running = false;
};
