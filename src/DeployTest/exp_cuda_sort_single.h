#pragma once
#include "cudaExp.h"

class CudaSortSingle : public CudaExperiment {
public:
  CudaSortSingle();
  ~CudaSortSingle();
  unsigned int GetMinCu();
  unsigned int GetMax();
  void Init(std::vector<cuda::CudaDevice> &devices);
  void Shutdown();

private:
  void Start(unsigned int num_runs, const std::vector<int> options);
};
