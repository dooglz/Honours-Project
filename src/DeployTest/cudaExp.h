#pragma once
#include <stdint.h>
#include <vector>
#include "Experiment.h"
#include "cuda_utils.h"
#include <mutex>
#include <thread>
#include <chrono>

using namespace std;

class CudaExperiment : public Experiment {
public:
  virtual void Init(std::vector<cuda::CudaDevice> &devices);
  virtual void Start(unsigned int num_runs, const std::vector<int> options) = 0;
  ~CudaExperiment();
  void Init2(bool batch, int selectedPlat, std::vector<int> selectedDevices);

protected:
  CudaExperiment(const unsigned int minCu, const unsigned int maxCU, const string &name,
                 const string &description);

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
