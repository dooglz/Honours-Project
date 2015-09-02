#include "sort.h"
#include <chrono> // std::chrono::seconds
#include <thread>
#include <stdio.h>

Sort::Sort() {}
Sort::~Sort() {}
uint16_t Sort::GetMinCu() { return 1; }
uint16_t Sort::GetMax() { return 4; }
void Sort::Init(cl_context &context, std::vector<cl::device> &devices, cl::platform platform) {}
void Sort::Shutdown() {}

void Sort::Work(uint16_t num_runs) {
  printf(" thread starting\n");
  uint16_t runs = 0;
  {
    std::lock_guard<std::mutex> lock(running_mutex);
    running = true;
  }
  while (ShouldRun() && runs < num_runs) {
    printf(" thread Working\n");
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  printf(" thread stopping\n");
  {
    std::lock_guard<std::mutex> lock(running_mutex);
    running = false;
  }
};