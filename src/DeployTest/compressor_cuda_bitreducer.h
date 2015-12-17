#pragma once
#include <stdint.h>
class compressor_cuda_bitreducer {
public:
  // Experiment();
  ~compressor_cuda_bitreducer();
  compressor_cuda_bitreducer();
  void Shutdown();
  void Init();
  void Compress(const uint32_t* gpuBuffer, const size_t dataSize);
  void Decompress();

protected:
};
