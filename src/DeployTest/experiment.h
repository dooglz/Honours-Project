#pragma once
#include <stdint.h>
#include <vector>
#include "opencl_utils.h"
#include <mutex>
#include <thread>
class Experiment {
public:
 // Experiment();
  ~Experiment();
  const uint16_t minCu;
  const uint16_t maxCU;
  const std::string name;
  const std::string description;
  virtual void Init(cl_context &context, std::vector<cl::device> &devices, cl::platform platform);
  virtual void Shutdown();
  bool IsRunning();
  bool ShouldRun();
  virtual void Start(uint16_t num_runs);
  virtual void Stop();

protected:
  Experiment(const uint16_t minCu, const uint16_t maxCU, const std::string name,
             const std::string description);
  std::thread workThread;
  virtual void Work(uint16_t num_runs) = 0;
  bool should_run;
  bool running;
  std::mutex should_run_mutex; // protects should_run
  std::mutex running_mutex;    // protects running
};
