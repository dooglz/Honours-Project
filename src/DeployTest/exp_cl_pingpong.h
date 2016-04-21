#pragma once
#include "opencl_exp.h"

class CL_PingPong : public OpenCLExperiment {
public:
	CL_PingPong();
	~CL_PingPong();
  unsigned int GetMinCu();
  unsigned int GetMax();
  void Init(cl_context &context, std::vector<cl_command_queue> &cq,
            std::vector<cl::CLDevice> &devices, cl::Platform platform);
  void Shutdown();

private:
  void Start(unsigned int num_runs, const std::vector<int> options);
};
