#pragma once
#include "experiment.h"


class CudaSort : public Experiment {
public:
  CudaSort();
  ~CudaSort();
  unsigned int GetMinCu();
  unsigned int GetMax();
  void Init(cl_context &context, std::vector<cl_command_queue> &cq,
            std::vector<cl::Device> &devices, cl::Platform platform);
  void Shutdown();

private:
  void Start(unsigned int num_runs, const std::vector<int> options);
};
