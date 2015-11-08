#pragma once
#include "experiment.h"

class Sort : public Experiment {
public:
  Sort();
  ~Sort();
  unsigned int GetMinCu();
  unsigned int GetMax();
  void Init(cl_context &context, std::vector<cl_command_queue> &cq,
    std::vector<cl::CLDevice> &devices, cl::Platform platform);
  void Shutdown();

private:
  void Start(unsigned int num_runs, const std::vector<int> options);
};
