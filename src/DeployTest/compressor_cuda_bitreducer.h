#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <vector>

class compressor_cuda_bitreducer {
public:
  // Experiment();
  ~compressor_cuda_bitreducer();
  compressor_cuda_bitreducer();
  void Shutdown();
  void Init();
  void Compress(const uint32_t* gpuBuffer, const size_t dataSize, uint32_t* &outBuffer, uint32_t &outSize, cudaStream_t &stream);
  void Decompress();

  //Note: will block when recording times
  void EnableTiming(std::vector<unsigned long long>* vtimes);
  void DisableTiming();
  std::vector<std::string>TimingHeadders();
protected:
	std::vector<unsigned long long>* times;
	bool timing;
};
