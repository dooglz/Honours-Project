#pragma once
#include "opencl_exp.h"

class Sort2 : public OpenCLExperiment {
public:
  Sort2();
  ~Sort2();
  unsigned int GetMinCu();
  unsigned int GetMax();
  void Init(cl_context &context, std::vector<cl_command_queue> &cq,
            std::vector<cl::CLDevice> &devices, cl::Platform platform);
  void Shutdown();

private:
  void Start(unsigned int num_runs, const std::vector<int> options);
};
