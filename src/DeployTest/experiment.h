#pragma once
#include <stdint.h>
#include <vector>
#include "opencl_utils.h"
#include <mutex>
#include <thread>
#include <chrono>

using namespace std;

class Experiment {
public:
  // Experiment();
  ~Experiment();
  const unsigned int minCu;
  const unsigned int maxCU;
  const std::string name;
  const std::string description;
  virtual void Shutdown();
  bool IsRunning();
  bool ShouldRun();
  virtual void Init2(bool batch, int selectedPlat, std::vector<int> selectedDevices) = 0;
  virtual void Start(unsigned int num_runs, const std::vector<int> options) = 0;
  virtual void Stop();

protected:
  Experiment(const unsigned int minCu, const unsigned int maxCU, const string &name,
             const string &description);

  bool should_run;
  bool running;

  template <typename T>
  const bool CheckArrayOrder(const T *a, const size_t size, const bool order) {
    if (size < 1) {
      return true;
    }
    for (size_t i = 1; i < size; i++) {
      if ((order && a[i] < a[i - 1]) || (!order && a[i] > a[i - 1])) {
        return false;
      }
    }
    return true;
  }
  template <typename T> const bool CheckArraysEqual(const T *a, const T *b, const size_t size) {
    if (size < 1) {
      return true;
    }
    for (size_t i = 1; i < size; i++) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  }
};
