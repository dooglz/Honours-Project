#include "exp_cl_pingpong.h"
#include "timer.h"
#include "utils.h"
#include <assert.h>
#include <chrono> // std::chrono::seconds
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <thread>

#define DEFAULTPOWER 18
#define VERIFY 0
CL_PingPong::CL_PingPong() : OpenCLExperiment(1, 4, "OpenCLPingPong", "Plays tennis with data") {}

CL_PingPong::~CL_PingPong() {}

static cl_context ctx;
static vector<cl::CLDevice> CtxDevices;
static std::vector<cl_command_queue> cq;
unsigned int CL_PingPong::GetMinCu() { return 2; }
unsigned int CL_PingPong::GetMax() { return 2; }
void CL_PingPong::Init(cl_context &context, std::vector<cl_command_queue> &commandQ,
                       std::vector<cl::CLDevice> &devices, cl::Platform platform) {
  CtxDevices = devices;
  ctx = context;
  cq = commandQ;
}
void CL_PingPong::Shutdown() {}

void CL_PingPong::Start(unsigned int num_runs, const std::vector<int> options) {
  int ret = 0;
  int wg = 256;
  auto tid = this_thread::get_id();
  std::cout << DASH50 << "\n Ping Pong Test\n";

  // decode options
  uint16_t power;
  if (options.size() > 0) {
    power = options[0];
  } else {
    cout << "Power of numbers to swap?: (0 for default)" << std::endl;
    power = promptValidated<int, int>("Power: ", [](int i) { return (i >= 0 && i <= 256); });
  }
  if (power == 0) {
    power = DEFAULTPOWER;
  }

  // load a pointelss kernel
  auto prog = cl::load_program("hello.cl", ctx, CtxDevices);
  cl_kernel kernel = clCreateKernel(prog, "pointless", &ret);
  assert(ret == CL_SUCCESS);

  /* Create  Random Numbers */
  cl_uint maxN = 1 << power;
  size_t szPC = maxN * sizeof(cl_uint);
  cl_uint *rndData = new cl_uint[maxN];

  // create buffers
  cl_mem buf1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, szPC, NULL, &ret);
  cl_mem buf2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, szPC, NULL, &ret);

  // try to place on correct gpus, broken AF
  // clEnqueueMigrateMemObjects(cq[0], 1, &buf1, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, NULL,
  // NULL, NULL);
  // clEnqueueMigrateMemObjects(cq[1], 1, &buf2, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, NULL,
  // NULL, NULL);

  // send data to buf 1
  ret = clEnqueueWriteBuffer(cq[0], buf1, CL_TRUE, 0, szPC, &rndData[0], 0, NULL, NULL); // blocking
  ret = clFinish(cq[0]); // Wait untill all commands executed.
  assert(ret == CL_SUCCESS);

  unsigned int runs = 0;
  running = true;
  should_run = true;
  ResultFile r;
  r.name = "CLPingPong" + to_string(maxN);
  r.headdings = {"A to B", "B to A"};

  while (ShouldRun() && runs < num_runs) {
    vector<unsigned long long> times;
    unsigned int percentDone = (unsigned int)(floor(((float)runs / (float)num_runs) * 100.0f));
    cout << "\r" << Spinner(runs) << "\t" << runs << "\tPercent Done: " << percentDone << "%"
         << std::flush;

    // make new numbers
    for (cl_uint i = 0; i < maxN; i++) {
      cl_uint x = (cl_uint)0;
      rndData[i] = (x << 14) | ((cl_uint)rand() & 0x3FFF);
      rndData[i] = (x << 14) | ((cl_uint)rand() & 0x3FFF);
    }

    Timer time_swap_a;
    ret = clEnqueueWriteBuffer(cq[1], buf2, CL_TRUE, 0, szPC, buf1, 0, NULL, NULL); // blocking
    for (auto q : cq) {
      ret = clFinish(q); // Wait untill all commands executed.
      assert(ret == CL_SUCCESS);
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)(&buf2));
    assert(ret == CL_SUCCESS);

    size_t global_work_size[1] = {maxN};
    ret = clEnqueueNDRangeKernel(cq[1], kernel,
                                 1,                // work_dim
                                 0,                // global_work_offset
                                 global_work_size, // global_work_size
                                 NULL,             // local_work_size
                                 0,                // num_events_in_wait_list
                                 NULL,             // event_wait_list
                                 NULL              // event
                                 );
    for (auto q : cq) {
      ret = clFinish(q); // Wait untill all commands executed.
      assert(ret == CL_SUCCESS);
    }

    time_swap_a.Stop();
    times.push_back(time_swap_a.Duration_NS());

    Timer time_swap_b;

    ret = clEnqueueWriteBuffer(cq[0], buf1, CL_TRUE, 0, szPC, buf2, 0, NULL, NULL); // blocking
    for (auto q : cq) {
      ret = clFinish(q); // Wait untill all commands executed.
      assert(ret == CL_SUCCESS);
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)(&buf1));
    assert(ret == CL_SUCCESS);

    ret = clEnqueueNDRangeKernel(cq[0], kernel,
                                 1,                // work_dim
                                 0,                // global_work_offset
                                 global_work_size, // global_work_size
                                 NULL,             // local_work_size
                                 0,                // num_events_in_wait_list
                                 NULL,             // event_wait_list
                                 NULL              // event
                                 );

    for (auto q : cq) {
      ret = clFinish(q); // Wait untill all commands executed.
      assert(ret == CL_SUCCESS);
    }

    time_swap_b.Stop();
    times.push_back(time_swap_b.Duration_NS());
    r.times.push_back(times);
    ++runs;
  }

  delete[] rndData;
  r.CalcAvg();
  r.PrintToCSV(r.name);
  cout << "\n PingPong finished\n";
  running = false;
};
