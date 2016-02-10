#pragma once
#include "compressor_cuda_bitreducer.h"
#include "cuda_utils.h"
#include "timer.h"
#include <vector>
#include <chrono>

#define COMPACTMODE 1

compressor_cuda_bitreducer::compressor_cuda_bitreducer() {}
compressor_cuda_bitreducer::~compressor_cuda_bitreducer() {}

void compressor_cuda_bitreducer::Init() { times = nullptr;  timing = false; }
void compressor_cuda_bitreducer::Shutdown() {}

void compressor_cuda_bitreducer::EnableTiming(std::vector<unsigned long long> *vtimes) {
  timing = true;
  times = vtimes;
}
void compressor_cuda_bitreducer::DisableTiming() {
  timing = false;
  times = nullptr;
}

std::vector<std::string> compressor_cuda_bitreducer::TimingHeadders() {
  return {"malloc", "reduce", "compact", "copyback", "ratio"};
}

void Run_BitReduce(const dim3 a, const dim3 b, const uint32_t *input_array, uint32_t *intBuf,
                   cudaStream_t &stream);
void Run_BitReduce_count(const dim3 a, const dim3 b, const uint32_t *input_array, uint32_t *intBuf,
	uint32_t *countBuf, cudaStream_t &stream);
void Run_seq_compact(const uint32_t *input_array, const uint16_t dataSize, uint32_t *sizeBuf,
                     cudaStream_t &stream);

void Run_count_compact(const uint32_t count, const uint32_t blockSize, uint32_t *intBuf, uint32_t *countBuf, cudaStream_t &stream);

void Run_move(uint8_t *buf, uint32_t dest, uint32_t source, uint16_t threads, uint16_t bytesEach, const bool wipe, cudaStream_t &stream);

void compressor_cuda_bitreducer::Compress(const uint32_t *gpuBuffer, const size_t dataSize,
                                          uint32_t *&outBuffer, uint32_t &outSize,
                                          cudaStream_t &stream) {

  cudaEvent_t events[5];
  if (timing) {
    for (auto &e : events) {
      cudaEventCreate(&e);
    }
    checkCudaErrors(cudaEventRecord(events[0], stream));
  }

  // Figure out our dimensions
  const unsigned int threadsperBlock = cuda::getBlockCount(512, dataSize);
  const dim3 blocks((dataSize / threadsperBlock), 1);
  const dim3 threads(threadsperBlock, 1);

  // create output
  const size_t int_buf_size = (dataSize * sizeof(uint32_t)) + (dataSize * sizeof(uint8_t));
  checkCudaErrors(cudaMalloc(&outBuffer, int_buf_size));
  uint32_t *sizeBuf;
  checkCudaErrors(cudaMalloc(&sizeBuf, 4));

#if COMPACTMODE == 1
  uint32_t *countBuffer;
  uint32_t *countBufferHost;
  uint8_t *outBufferHost;
  checkCudaErrors(cudaMalloc(&countBuffer, blocks.x * sizeof(uint32_t)));
  checkCudaErrors(cudaMallocHost(&outBufferHost, int_buf_size));
  checkCudaErrors(cudaMallocHost((void**)&countBufferHost, blocks.x * sizeof(uint32_t)));
#endif

  if (timing) {
    checkCudaErrors(cudaEventRecord(events[1], stream));
  }

// reduce the bits
#if COMPACTMODE == 0
  Run_BitReduce(blocks, threads, gpuBuffer, outBuffer, stream);
  getLastCudaError("Run_BitReduce() execution failed.\n");
#else
  Run_BitReduce_count(blocks, threads, gpuBuffer, outBuffer, countBuffer, stream);
  getLastCudaError("Run_BitReduce_count() execution failed.\n");
#endif

  if (timing) {
    checkCudaErrors(cudaEventRecord(events[2], stream));
  }

#if COMPACTMODE == 0
  // compact
  Run_seq_compact(outBuffer, dataSize, sizeBuf, stream);
  getLastCudaError("Run_seq_compact() execution failed.\n");

#else

  cudaMemcpyAsync(countBufferHost, countBuffer, blocks.x * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
  //cudaMemcpy(outBufferHost, outBuffer, int_buf_size, cudaMemcpyDeviceToHost);
  //cudaDeviceSynchronize();

  checkCudaErrors(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < blocks.x - 1; i++)
  {
	  uint32_t count = countBufferHost[0];
	  for (uint32_t j = 0; j < i; j++)
	  {
		  count += countBufferHost[i];
	  }
	  count += ((i + 1) * threads.x);

	  const uint32_t writeAddress = count;
	  const uint32_t readAddress = (i + 1) *(sizeof(uint32_t) * threads.x);
	  const uint32_t compactedBlockSize = countBufferHost[i + 1] + threads.x;
	  const uint32_t threadsToRun = cuda::getBlockCount(512, compactedBlockSize);
	  const uint32_t bytesToCopyPerThread = compactedBlockSize / threadsToRun;
	  Run_move((uint8_t *)outBuffer, writeAddress, readAddress, threadsToRun, bytesToCopyPerThread, true, stream);

	  //cudaDeviceSynchronize();

	  if (i == (blocks.x - 2)){
		  outSize = writeAddress + compactedBlockSize;
	  }
  }
  cudaMemcpy(outBufferHost, outBuffer, int_buf_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

#endif

  if (timing) {
    checkCudaErrors(cudaEventRecord(events[3], stream));
  }

#if COMPACTMODE == 0
  checkCudaErrors(cudaMemcpyAsync(&outSize, sizeBuf, 4, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaFree(sizeBuf));
#else
  checkCudaErrors(cudaFree(countBuffer));
  checkCudaErrors(cudaFreeHost(countBufferHost));
  checkCudaErrors(cudaFreeHost(outBufferHost));
#endif

  if (timing) {
    checkCudaErrors(cudaEventRecord(events[4], stream));
    // we have to block to enable timings
    checkCudaErrors(cudaStreamSynchronize(stream));

    for (size_t i = 1; i < 5; i++) {
      float time_ms;
      checkCudaErrors(cudaEventElapsedTime(&time_ms, events[i - 1], events[i]));
      times->push_back(msFloatTimetoNS(time_ms));
    }
    for (auto e : events) {
      cudaEventDestroy(e);
    }
  }

  const float ratio = (float)outSize / (float)(dataSize * sizeof(uint32_t));
  if (timing) {
    times->push_back((unsigned long long)(ratio*10000.0f));
  }
}

void compressor_cuda_bitreducer::Decompress() {}
