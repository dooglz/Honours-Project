#pragma once
#include "cudaExp.h"

  class Exp_Cuda_GFC : public CudaExperiment {
  public:
    Exp_Cuda_GFC();
    ~Exp_Cuda_GFC();
    unsigned int GetMinCu();
    unsigned int GetMax();
    void Init(std::vector<cuda::CudaDevice> &devices);
    void Shutdown();

  private:
    void Start(unsigned int num_runs, const std::vector<int> options);
  };
