#pragma once
#include "experiment.h"

class Sort: public Experiment {
public:
  Sort();
  ~Sort();
  uint16_t GetMinCu();
  uint16_t GetMax();
  void Init(cl_context &context, std::vector<cl::device> &devices, cl::platform platform);
  void Shutdown();

private:
  void Work(uint16_t num_runs);

};
