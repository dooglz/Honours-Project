#include "sort.h"
#include "utils.h"
#include <chrono> // std::chrono::seconds
#include <thread>
#include <stdio.h>
#include <iostream>
#include <math.h>
using namespace std;
Sort::Sort() : Experiment(1, 4, "Sort", "Sorts Things") {}

Sort::~Sort() {}
uint16_t Sort::GetMinCu() { return 1; }
uint16_t Sort::GetMax() { return 4; }
void Sort::Init(cl_context &context, vector<cl::device> &devices, cl::platform platform) {}
void Sort::Shutdown() {}

void Sort::Work(uint16_t num_runs) {
  auto tid = this_thread::get_id();
  std::cout << DASH50 << "\n Sort Test, Thread(" << tid << ")\n";
  uint16_t runs = 0;
  {
    std::lock_guard<std::mutex> lock(running_mutex);
    running = true;
  }
  while (ShouldRun() && runs < num_runs) {
    // printf(" thread Working\n");
    uint8_t percentDone = floor(((float)runs / (float)num_runs) * 100.0f);
    wprintf(L" %c\t%u\t%Percent Done: %u%%  \t\t\r", Spinner(runs), runs, percentDone);
    // if (runs % 5 == 0){ std::cout << "\r"; }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ++runs;
  }
  printf("\n thread stopping\n");
  {
    std::lock_guard<std::mutex> lock(running_mutex);
    running = false;
  }
};
