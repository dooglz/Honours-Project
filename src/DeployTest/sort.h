#pragma once
#include "experiment.h"

class Sort : public Experiment {
public:
  Sort();
  ~Sort();
  unsigned int GetMinCu();
  unsigned int GetMax();
  void Init(cl_context &context, cl_command_queue &cq, std::vector<cl::device> &devices, cl::platform platform);
  void Shutdown();

private:
  void Work(unsigned int num_runs);
};
