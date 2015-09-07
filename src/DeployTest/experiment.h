#pragma once
#include <stdint.h>
#include <vector>
#include "opencl_utils.h"
#include <mutex>
#include <thread>
#include <chrono>

using namespace std;

struct Timer {
  chrono::steady_clock::time_point start;
  chrono::steady_clock::time_point end;
  string name;
  Timer() { Start(); }
  Timer(const string &n) {
    name = n;
    Start();
  }
  void Start() { start = chrono::steady_clock::now(); }
  void Stop() { end = chrono::steady_clock::now(); }
  const chrono::steady_clock::duration Duration() { return end - start; }
};
class Experiment {
public:
  // Experiment();
  ~Experiment();
  const unsigned int minCu;
  const unsigned int maxCU;
  const std::string name;
  const std::string description;
  virtual void Init(const cl_context &ctx, const cl_command_queue &cq,
                    const std::vector<cl::device> &devices, const cl::platform platform);
  virtual void Shutdown();
  bool IsRunning();
  bool ShouldRun();
  virtual void Start(unsigned int num_runs);
  virtual void Stop();

protected:
  Experiment(const unsigned int minCu, const unsigned int maxCU, const string &name,
             const string &description);
  thread workThread;
  virtual void Work(unsigned int num_runs) = 0;
  bool should_run;
  bool running;
  mutex should_run_mutex; // protects should_run
  mutex running_mutex;    // protects running

  // print funcs
  const void Experiment::PrintToCSV(const string &collumn1, const string &collumn2,
                                    const vector<Timer> &times, const string &filename);
  const void Experiment::PrintToCSV(const vector<vector<string>> v, const string &filename);
};
