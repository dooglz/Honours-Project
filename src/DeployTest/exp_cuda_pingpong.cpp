#include "timer.h"
#include "exp_cuda_pingpong.h"
#include "utils.h"
#include <assert.h>
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <thread>

//#include <cuda.h>
#include <cuda_runtime.h>

#define DEFAULTPOWER 18

Exp_Cuda_PingPong::Exp_Cuda_PingPong()
    : CudaExperiment(1, 4, "CudaPingPong", "Plays tennis with data") {}
Exp_Cuda_PingPong::~Exp_Cuda_PingPong() {}

static vector<cuda::CudaDevice> CtxDevices;
unsigned int Exp_Cuda_PingPong::GetMinCu() { return 2; }
unsigned int Exp_Cuda_PingPong::GetMax() { return 2; }

void Run_NothingKernel(dim3 a, dim3 b, cudaStream_t stream, uint32_t*buf);

void Exp_Cuda_PingPong::Init(std::vector<cuda::CudaDevice> &devices) { CtxDevices = devices; }
void Exp_Cuda_PingPong::Shutdown() {}

void Exp_Cuda_PingPong::Start(unsigned int num_runs, const std::vector<int> options) {
  if (CtxDevices.size() < GetMinCu() || CtxDevices.size() > GetMax()) {
    std::cout << "\n invalid number of devices\n";
    return;
  }
  std::cout << "\n CudaPingPong\n";
  // decode options
  uint16_t power;
  if (options.size() > 0) {
    power = options[0];
  } else {
    cout << "Power of numbers to swap?: (0 for default)" << std::endl;
    power = promptValidated<int, int>("Power: ", [](int i) { return (i >= 0 && i <= 256); });
  }
  if (power == 0) {
    power = DEFAULTPOWER;
  }
  int optmode = 0;
  if (options.size() > 1) {
    optmode = options[1];
  } else {
    cout << "Data transfer mode (0 HostRam, 1 peer, 2 UVA)" << std::endl;
    optmode = promptValidated<int, int>("optmode: ", [](int i) { return (i >= 0 && i <= 2); });
  }
  if (optmode == 1) {
    if (!cuda::enableP2P(0, 1)) {
      cerr << "Couldn't enable P2P, returning" << std::endl;
      return;
    }
  } else if (optmode == 2) {
    cerr << "Couldn't enable UVA, returning" << std::endl;
    if (!cuda::enableUVA(0, 1)) {
      return;
    }
  }

  const uint32_t count = 1 << power;
  const size_t dataSize = (count) * sizeof(uint32_t);

  uint32_t *device_mem[2];
  cudaStream_t streams[2];
  uint32_t *host_mem;

  unsigned int threadsperBlock = cuda::getBlockCount(512, count);
  dim3 blocks((count / threadsperBlock), 1);
  dim3 threads(threadsperBlock, 1);

  // malloc
  checkCudaErrors(cudaMallocHost((void **)&host_mem, dataSize));
  for (size_t i = 0; i < 2; i++) {
    checkCudaErrors(cudaSetDevice(CtxDevices[i].id));
    checkCudaErrors(cudaMalloc(&device_mem[i], dataSize));
    checkCudaErrors(cudaStreamCreate(&streams[i]));
  }

  // gen data
  for (size_t i = 0; i < count; ++i) {
    uint32_t x = 0;
    host_mem[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
    host_mem[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
  }

  // send data to gpu0
  checkCudaErrors(cudaSetDevice(CtxDevices[0].id));
  checkCudaErrors(
      cudaMemcpyAsync(device_mem[0], host_mem, dataSize, cudaMemcpyHostToDevice, streams[0]));
  checkCudaErrors(cudaDeviceSynchronize());

  unsigned int runs = 0;
  running = true;
  should_run = true;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  ResultFile r;
  float time_ms;
  r.name = "Cuda_PingPong_" + to_string(optmode)+ "_" + to_string(count);
  r.headdings = {"A to B", "B to A"};

  while (ShouldRun() && runs < num_runs) {
    vector<unsigned long long> times;
    unsigned int percentDone = (unsigned int)(floor(((float)runs / (float)num_runs) * 100.0f));
    cout << "\r" << Spinner(runs) << "\t" << runs << "\tPercent Done: " << percentDone << "%"
         << std::flush;
    // copy from 0 to 1
    checkCudaErrors(cudaEventRecord(start, streams[0]));
    switch (optmode) {
    case 0: // copy to host
      checkCudaErrors(
          cudaMemcpyAsync(host_mem, device_mem[0], dataSize, cudaMemcpyDeviceToHost, streams[0]));
      checkCudaErrors(
          cudaMemcpyAsync(device_mem[1], host_mem, dataSize, cudaMemcpyHostToDevice, streams[0]));
      break;
    case 1: // use peercopy
      checkCudaErrors(cudaMemcpyPeer(device_mem[1], 1, device_mem[0], 0, dataSize));
      break;
    case 2: // use UVA
      checkCudaErrors(
          cudaMemcpyAsync(device_mem[1], device_mem[0], dataSize, cudaMemcpyDefault, streams[0]));
      break;
    }
    checkCudaErrors(cudaEventRecord(end, streams[0]));

    checkCudaErrors(cudaStreamSynchronize(streams[0]));
    checkCudaErrors(cudaEventElapsedTime(&time_ms, start, end));
    times.push_back(msFloatTimetoNS(time_ms));


    //Run a junk kernel just to force the driver to make sure stuff actually copied
    checkCudaErrors(cudaSetDevice(CtxDevices[1].id));
    Run_NothingKernel(blocks, threads, streams[0], device_mem[1]);
    cudaStreamSynchronize(streams[0]);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaSetDevice(CtxDevices[0].id));
    cudaDeviceSynchronize();

    // copy back
    checkCudaErrors(cudaEventRecord(start, streams[0]));
    switch (optmode) {
    case 0: // copy to host
      checkCudaErrors(
          cudaMemcpyAsync(host_mem, device_mem[1], dataSize, cudaMemcpyDeviceToHost, streams[0]));
      checkCudaErrors(
          cudaMemcpyAsync(device_mem[0], host_mem, dataSize, cudaMemcpyHostToDevice, streams[0]));
      break;
    case 1: // use peercopy
      checkCudaErrors(cudaMemcpyPeer(device_mem[0], 0, device_mem[1], 1, dataSize));
      break;
    case 2: // use UVA
      checkCudaErrors(
          cudaMemcpyAsync(device_mem[0], device_mem[1], dataSize, cudaMemcpyDefault, streams[0]));
      break;
    }
    checkCudaErrors(cudaEventRecord(end, streams[0]));

    //Run a junk kernel just to force the driver to make sure stuff actually copied
    checkCudaErrors(cudaSetDevice(CtxDevices[0].id));
    Run_NothingKernel(blocks, threads, streams[0], device_mem[0]);
    cudaStreamSynchronize(streams[0]);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaSetDevice(CtxDevices[1].id));
    cudaDeviceSynchronize();

    checkCudaErrors(cudaStreamSynchronize(streams[0]));
    checkCudaErrors(cudaEventElapsedTime(&time_ms, start, end));
    times.push_back(msFloatTimetoNS(time_ms));

    r.times.push_back(times);
    ++runs;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  checkCudaErrors(cudaSetDevice(CtxDevices[0].id));
  checkCudaErrors(cudaFree(device_mem[0]));
  checkCudaErrors(cudaSetDevice(CtxDevices[1].id));
  checkCudaErrors(cudaFree(device_mem[1]));
  checkCudaErrors(cudaFreeHost(host_mem));
  checkCudaErrors(cudaStreamDestroy(streams[0]));
  checkCudaErrors(cudaStreamDestroy(streams[1]));

  r.CalcAvg();
  r.PrintToCSV(r.name);
  cout << "\n Exp finished\n";
  running = false;
};
