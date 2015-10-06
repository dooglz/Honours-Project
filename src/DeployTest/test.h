#pragma once
#include "experiment.h"

class Test : public Experiment {
public:
  Test();
  ~Test();
  unsigned int GetMinCu();
  unsigned int GetMax();
  void Init(cl_context &context, std::vector<cl_command_queue>  &cq, std::vector<cl::Device> &devices, cl::Platform platform);
  void Shutdown();

private:
  void Work(unsigned int num_runs);
};
