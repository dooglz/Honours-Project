#pragma once
#include "experiment.h"

class Sort : public Experiment {
public:
  Sort();
  ~Sort();
  unsigned int GetMinCu();
  unsigned int GetMax();
  void Init(cl_context &context, std::vector<cl::device> &devices, cl::platform platform);
  void Shutdown();

private:
  void Work(unsigned int num_runs);
};
