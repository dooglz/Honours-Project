#include "sort.h"
#include "utils.h"
#include <chrono> // std::chrono::seconds
#include <thread>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#define MEM_SIZE (128)

Sort::Sort() : Experiment(1, 4, "Sort", "Sorts Things") {}

Sort::~Sort() {}

static cl_context ctx;
static vector<cl::Device> CtxDevices;
static cl_command_queue cq;
unsigned int Sort::GetMinCu() { return 1; }
unsigned int Sort::GetMax() { return 4; }
void Sort::Init(cl_context &context, cl_command_queue &commandQ, std::vector<cl::Device> &devices,
                cl::Platform platform) {
  CtxDevices = devices;
  ctx = context;
  cq = commandQ;
}
void Sort::Shutdown() {}

void Sort::Work(unsigned int num_runs) {
  auto tid = this_thread::get_id();
  std::cout << DASH50 << "\n Sort Test, Thread(" << tid << ")\n";
  char outstring[MEM_SIZE];
  auto prog = cl::load_program("hello.cl", ctx, CtxDevices[0].id, 1);
  /* Create OpenCL Kernel */
  cl_int ret;
  auto kernel = clCreateKernel(prog, "hello", &ret);
  assert(ret == CL_SUCCESS);
  /* Create Memory Buffer */
  auto memobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(char), NULL, &ret);
  assert(ret == CL_SUCCESS);
  /* Set OpenCL Kernel Parameters */
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);
  assert(ret == CL_SUCCESS);
  /* Execute OpenCL Kernel */
  ret = clEnqueueTask(cq, kernel, 0, NULL, NULL);
  assert(ret == CL_SUCCESS);
  /* Copy results from the memory buffer */
  ret = clEnqueueReadBuffer(cq, memobj, CL_TRUE, 0, MEM_SIZE * sizeof(char), outstring, 0, NULL,
                            NULL);
  assert(ret == CL_SUCCESS);

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
