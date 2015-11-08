#include "test.h"
#include "utils.h"
#include <chrono> // std::chrono::seconds
#include <thread>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#define MEM_SIZE (128)

Test::Test() : Experiment(1, 4, "Test", "Test Kernel Execution") {}

Test::~Test() {}

static cl_context ctx;
static vector<cl::CLDevice> CtxDevices;
static std::vector<cl_command_queue> cq;
unsigned int Test::GetMinCu() { return 1; }
unsigned int Test::GetMax() { return 4; }
void Test::Init(cl_context &context, std::vector<cl_command_queue> &commandQ,
  std::vector<cl::CLDevice> &devices, cl::Platform platform) {
  CtxDevices = devices;
  ctx = context;
  cq = commandQ;
}
void Test::Shutdown() {}

void Test::Start(unsigned int num_runs, const std::vector<int> options) {
  auto tid = this_thread::get_id();
  std::cout << DASH50 << "\n Test Test, Thread(" << tid << ")\n";
  char outstring[MEM_SIZE];
  auto prog = cl::load_program("hello.cl", ctx, CtxDevices);
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

  unsigned int runs = 0;
  running = true;

  while (ShouldRun() && runs < num_runs) {

    unsigned int percentDone = (unsigned int)(floor(((float)runs / (float)num_runs) * 100.0f));
    cout << "\r" << Spinner(runs) << "\t" << runs << "\tPercent Done: " << percentDone << "%" << std::flush;
    //
    /* Execute OpenCL Kernel */
    ret = clEnqueueTask(cq[0], kernel, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);
    /* Copy results from the memory buffer */
    ret = clEnqueueReadBuffer(cq[0], memobj, CL_TRUE, 0, MEM_SIZE * sizeof(char), outstring, 0,
                              NULL, NULL);
    assert(ret == CL_SUCCESS);
    string s = outstring;
    assert(s == "Hello, World!");
    //
    ++runs;
  }
  printf("\n thread stopping\n");
  running = false;
};
