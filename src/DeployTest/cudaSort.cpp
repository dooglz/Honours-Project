#include "cudaSort.h"
#include "cuda_utils.h"
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
void Runbitonic_sort_step(dim3 a, dim3 b, float *dev_values, int j, int k;
void RunSortKernel(dim3 blocks, dim3 threads, int *theArray, const unsigned int stage,
  const unsigned int passOfStage, const unsigned int width);

#define DEFAULTPOWER 18
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
const int threadsperblock = 512;
uint32_t mxount =0;
void CudaSort::Start(unsigned int num_runs, const std::vector<int> options) {
  cout << "\n cuda Sort\n";
  const int GPU_N = cuda::total_num_devices;
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

  cudaStream_t *streams = new cudaStream_t[GPU_N];
  uint32_t **inBuffers = new uint32_t *[GPU_N];
  uint32_t **hostBuffers = new uint32_t *[GPU_N];
  // Create streams for issuing GPU command asynchronously and allocate memory (GPU and System
  // page-locked)
  for (auto i = 0; i < GPU_N; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaStreamCreate(&streams[i]));
    // Allocate memory
    checkCudaErrors(cudaMalloc((void **)&inBuffers[i], szPC));
    checkCudaErrors(cudaMallocHost((void **)&hostBuffers[i], szPC));

    // move rnd numbers into gpu readable host memory
    for (auto j = 0; j < maxNPC; j++) {
      uint32_t offset = (i * maxNPC);
      hostBuffers[i][j] = rndData[j + offset];
    }
  }

  // Copy data to GPU, launch the kernel and copy data back. All asynchronously
  for (auto i = 0; i < GPU_N; i++) {
    // Set device
    checkCudaErrors(cudaSetDevice(i));

    // Copy input data from CPU
    checkCudaErrors(
        cudaMemcpyAsync(inBuffers[i], hostBuffers[i], szPC, cudaMemcpyHostToDevice, streams[i]));
  }
  // wait
  for (auto i = 0; i < GPU_N; i++) {
    cudaStreamSynchronize(streams[i]);
  }
  //args emulation
  int ** arg_inputArray = new int*[GPU_N];
  unsigned int * arg_stage = new unsigned int[GPU_N];
  unsigned int * arg_passOfStage = new unsigned int[GPU_N];
  unsigned int * arg_width = new unsigned int[GPU_N];

  /*
  * 2^numStages should be equal to length.
  * i.e the number of times you halve length to get 1 should be numStages
  */
  int temp;
  uint32_t numStages = 0;
  for (temp = maxNPC; temp > 2; temp >>= 1) {
    ++numStages;
  }

  // run the sort.
  size_t nThreads[1];
  nThreads[0] = maxNPC / (2 * 4);

  unsigned int swapcount = 0;
  if (GPU_N == 2) {
    for (uint32_t swapsize = maxNPC / 2; swapsize > 0; swapsize /= 2) {
      for (uint32_t stage = 0; stage < numStages; stage++) {
        // stage of the algorithm
        for (size_t i = 0; i < GPU_N; i++) {
          arg_stage[i] = stage;
        }

        // Every stage has stage + 1 passes
        for (int passOfStage = stage; passOfStage >= 0; passOfStage--) {
          for (size_t i = 0; i < GPU_N; i++) {
            arg_passOfStage[i] = passOfStage;
          }

          size_t global_work_size[1] = {passOfStage ? nThreads[0] : nThreads[0] << 1};
          for (size_t i = 0; i < GPU_N; i++) {
            unsigned int threadsperBlock = cuda::getBlockCount(threadsperblock, global_work_size[0]);
            unsigned int blocks = global_work_size[0] / threadsperBlock;
            checkCudaErrors(cudaSetDevice(i));
            dim3 a( threadsperBlock, 1);
            dim3 b( blocks,1);
            RunSortKernel(b, a, (int*)inBuffers[i], arg_stage[i], arg_passOfStage[i], maxNPC);
            getLastCudaError("reduceKernel() execution failed.\n");


          }
          for (auto i = 0; i < GPU_N; i++) {
            cudaStreamSynchronize(streams[i]);
            getLastCudaError("reduceKernel() execution failed.\n");
          }
        }
      }

      //read back all data
      for (auto i = 0; i < GPU_N; i++)
      {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaMemcpyAsync(hostBuffers[i], inBuffers[i], szPC, cudaMemcpyDeviceToHost, streams[i]));
      }
      for (auto i = 0; i < GPU_N; i++) {
        cudaStreamSynchronize(streams[i]);
      }

      // read in the top of card 1 to temp
      uint32_t *tmpData = new uint32_t[swapsize];
      uint32_t a = swapsize * sizeof(uint32_t);
      std::copy(hostBuffers[1], &hostBuffers[1][swapsize], tmpData);

      // copy bottom of card 0 to top of card 1
      std::copy(&hostBuffers[0][(maxNPC - swapsize)], &hostBuffers[0][(maxNPC - swapsize)+a], hostBuffers[1]);
      // write the top of card 1 to the bottom of card 0
      std::copy(tmpData, &tmpData[swapsize], &hostBuffers[0][(maxNPC - swapsize)]);

      // Copy data back to GPU
      for (auto i = 0; i < GPU_N; i++) {
        // Set device
        checkCudaErrors(cudaSetDevice(i));

        // Copy input data from CPU
        checkCudaErrors(
          cudaMemcpyAsync(inBuffers[i], hostBuffers[i], szPC, cudaMemcpyHostToDevice, streams[i]));
      }
      // wait
      for (auto i = 0; i < GPU_N; i++) {
        cudaStreamSynchronize(streams[i]);
      }


      delete[] tmpData;
      ++swapcount;
    }
  }

  //read back all data
  for (auto i = 0; i < GPU_N; i++)
  {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaMemcpyAsync(hostBuffers[i], inBuffers[i], szPC, cudaMemcpyDeviceToHost, streams[i]));
  }
  for (auto i = 0; i < GPU_N; i++) {
    cudaStreamSynchronize(streams[i]);
  }
  cout << 22;

  /*
  Timer time_sort;
  // do one final sort, or possibly the first sort if 1 gpu
  for (cl_int stage = 0; stage < numStages; stage++) {
    // stage of the algorithm
    for (size_t i = 0; i < cq.size(); i++) {
      ret = clSetKernelArg(kernels[i], 1, sizeof(cl_uint), (void *)&stage);
      assert(ret == CL_SUCCESS);
    }
    // Every stage has stage + 1 passes
    for (cl_int passOfStage = stage; passOfStage >= 0; passOfStage--) {
      for (size_t i = 0; i < cq.size(); i++) {
        ret = clSetKernelArg(kernels[i], 2, sizeof(cl_uint), (void *)&passOfStage);
        assert(ret == CL_SUCCESS);
      }

      size_t global_work_size[1] = {passOfStage ? nThreads[0] : nThreads[0] << 1};

      for (size_t i = 0; i < cq.size(); i++) {
        ret =
            clEnqueueNDRangeKernel(cq[i], kernels[i], 1, 0, global_work_size, NULL, 0, NULL, &e[i]);
        assert(ret == CL_SUCCESS);
      }
      for (auto q : cq) {
        ret = clFinish(q); // Wait untill all commands executed.
        assert(ret == CL_SUCCESS);
      }
    }
  }
  time_sort.Stop();
  if (runs == 0) {
    r.headdings.push_back("Sort_" + to_string(swapcount));
  }
  times.push_back(time_sort.Duration_NS());
  if (cq.size() == 2) {
    // may be a pssobility that the 2 edge values arn't in the corrct place
    cl_uint a = 0;
    cl_uint b = 0;
    // last value of 0
    ret = clEnqueueReadBuffer(cq[0], inBuffers[0], CL_TRUE, szPC - sizeof(cl_uint), sizeof(cl_uint),
                              &a, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);
    // fist value of 1
    ret = clEnqueueReadBuffer(cq[1], inBuffers[1], CL_TRUE, 0, sizeof(cl_uint), &b, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);
    if (a < b) {
      // yup, swap them round
      ret = clEnqueueWriteBuffer(cq[0], inBuffers[0], CL_TRUE, szPC - sizeof(cl_uint),
                                 sizeof(cl_uint), &b, 0, NULL, NULL);
      assert(ret == CL_SUCCESS);
      ret =
          clEnqueueWriteBuffer(cq[1], inBuffers[1], CL_TRUE, 0, sizeof(cl_uint), &a, 0, NULL, NULL);
      assert(ret == CL_SUCCESS);
    }
  }
  // stop timer here

  Timer time_copyback;
  // Copy results from the memory buffer
  cl_uint *outData = new cl_uint[maxN];
  for (size_t i = 0; i < cq.size(); i++) {
    cl_uint offset = (i * maxNPC);
    ret =
        clEnqueueReadBuffer(cq[i], inBuffers[i], CL_TRUE, 0, szPC, &outData[offset], 0, NULL, NULL);
    assert(ret == CL_SUCCESS);
    ret = clFinish(cq[i]); // Wait untill all commands executed.
    assert(ret == CL_SUCCESS);
  }
  time_copyback.Stop();
#ifdef VERIFY
  assert(CheckArrayOrder(outData, maxN, false));
#endif
  delete outData;

  if (runs == 0) {
    r.headdings.push_back("Copy Back");
  }
  times.push_back(time_copyback.Duration_NS());
  r.times.push_back(times);
  ++runs;
}
*/
// Cleanup and shutdown
for (auto i = 0; i < GPU_N; i++) {
  checkCudaErrors(cudaSetDevice(i));
  checkCudaErrors(cudaFree(inBuffers[i]));
  checkCudaErrors(cudaStreamDestroy(streams[i]));
  checkCudaErrors(cudaFreeHost(hostBuffers[i]));
  cudaDeviceReset();
}

delete streams;
delete[] hostBuffers;
delete[] inBuffers;
delete[] rndData;

cout << "\n Sort finished\n";
running = false;
}
;
