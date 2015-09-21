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

  int wg = 32;
  auto tid = this_thread::get_id();
  std::cout << DASH50 << "\n Sort Test, Thread(" << tid << ")\n";
  char outstring[MEM_SIZE];
  auto prog = cl::load_program("sort.cl", ctx, CtxDevices[0].id, 1);

  /* Create OpenCL Kernel */
  cl_int ret;
  auto kernel = clCreateKernel(prog, "ParallelSelection_Local", &ret);
  assert(ret == CL_SUCCESS);

  /* Create Sapce for Random Numbers */
  int maxN = 1 << 24;
  cl_uint *rndData = new cl_uint[maxN];

  /*Assign memory*/
  size_t sz = maxN * sizeof(cl_uint);
  cl_mem inBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sz, NULL, &ret);
  assert(ret == CL_SUCCESS);
  cl_mem outBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sz, NULL, &ret);
  assert(ret == CL_SUCCESS);

  /* Set OpenCL Kernel Parameters */
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inBuffer);
  assert(ret == CL_SUCCESS);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outBuffer);
  assert(ret == CL_SUCCESS);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_uint) * wg, NULL);
  assert(ret == CL_SUCCESS);

  unsigned int runs = 0;
  {
    std::lock_guard<std::mutex> lock(running_mutex);
    running = true;
  }
  std::vector<Timer> times;
  while (ShouldRun() && runs < num_runs) {

    // printf(" thread Working\n");
    unsigned int percentDone = (unsigned int)(floor(((float)runs / (float)num_runs) * 100.0f));
    wprintf(L" %c\t%u\t%Percent Done: %u%%  \t\t\r", Spinner(runs), runs, percentDone);
    //

    // make new numbers
    for (int i = 0; i < maxN; i++) {
      cl_uint x = (cl_uint)0;
      rndData[i] = (x << 14) | ((cl_uint)rand() & 0x3FFF);
      rndData[i] = (x << 14) | ((cl_uint)rand() & 0x3FFF);
    }
    // send data
    ret = clEnqueueWriteBuffer(cq, inBuffer, CL_TRUE, 0, sz, rndData, 0, NULL, NULL); // blocking
    clFinish(cq); // Wait untill all commands executed.

    Timer t = Timer(to_string(runs));
    // run the sort.
    size_t nThreads[1];
    nThreads[0] = maxN;
    size_t workGroup[1];
    workGroup[0] = wg;
    cl_event e;
    ret = clEnqueueNDRangeKernel(cq, kernel, 1, 0, nThreads, workGroup, NULL, 0, &e);
    assert(ret == CL_SUCCESS);
    clFinish(cq); // Wait untill all commands executed.

    /*
    // Copy results from the memory buffer
    cl_uint *outData = new cl_uint[maxN];
    ret = clEnqueueReadBuffer(cq, outBuffer, CL_TRUE, 0, sz, outData, 0, NULL, NULL);
    clFinish(cq); // Wait untill all commands executed.
    */

    //
    ++runs;
    t.Stop();
    times.push_back(t);
  }
  printf("\n thread stopping\n");
  // delete outData;
  delete rndData;
  PrintToCSV("Run #", "Time", times, "sort_" + current_time_and_date());
  {

    std::lock_guard<std::mutex> lock(running_mutex);
    running = false;
  }
};
