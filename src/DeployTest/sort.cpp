#include "sort.h"
#include "utils.h"
#include <chrono> // std::chrono::seconds
#include <thread>
#include <stdio.h>
#include <iostream>
#include <math.h>

Sort::Sort() : Experiment(1, 4, "Sort", "Sorts Things") {}

Sort::~Sort() {}

static cl_context ctx;
static vector<cl::device> CtxDevices;
unsigned int Sort::GetMinCu() { return 1; }
unsigned int Sort::GetMax() { return 4; }
void Sort::Init(cl_context &context, vector<cl::device> &devices, cl::platform platform) { CtxDevices = devices; ctx = context; }
void Sort::Shutdown() {}

void Sort::Work(unsigned int num_runs) {
  auto tid = this_thread::get_id();
  std::cout << DASH50 << "\n Sort Test, Thread(" << tid << ")\n";

  auto prog = cl::load_program("hello.cl", ctx, CtxDevices[0].id, 0);

  unsigned int runs = 0;
  {
    std::lock_guard<std::mutex> lock(running_mutex);
    running = true;
  }
  std::vector<Timer> times;
  while (ShouldRun() && runs < num_runs) {
    Timer t = Timer(to_string(runs));
    // printf(" thread Working\n");
    unsigned int percentDone = (unsigned int)(floor(((float)runs / (float)num_runs) * 100.0f));
    wprintf(L" %c\t%u\t%Percent Done: %u%%  \t\t\r", Spinner(runs), runs, percentDone);
    // if (runs % 5 == 0){ std::cout << "\r"; }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    ++runs;
    t.Stop();
    times.push_back(t);
  }
  printf("\n thread stopping\n");
  {
    PrintToCSV("Run #", "Time", times, "sort_" + current_time_and_date());
    std::lock_guard<std::mutex> lock(running_mutex);
    running = false;
  }
};
