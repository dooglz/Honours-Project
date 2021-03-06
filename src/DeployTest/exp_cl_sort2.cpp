#include "exp_cl_sort2.h"
#include "utils.h"
#include "timer.h"
#include <chrono> // std::chrono::seconds
#include <thread>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <assert.h>

#define DEFAULTPOWER 12
#define VERIFY 0
Sort2::Sort2() : OpenCLExperiment(1, 4, "Sort2", "Sorts Things, Fast Bitonic") {}

Sort2::~Sort2() {}

static cl_context ctx;
static vector<cl::CLDevice> CtxDevices;
static std::vector<cl_command_queue> cq;
unsigned int Sort2::GetMinCu() { return 1; }
unsigned int Sort2::GetMax() { return 4; }
void Sort2::Init(cl_context &context, std::vector<cl_command_queue> &commandQ,
                 std::vector<cl::CLDevice> &devices, cl::Platform platform) {
  CtxDevices = devices;
  ctx = context;
  cq = commandQ;
}
void Sort2::Shutdown() {}

void Sort(int size, std::vector<cl_kernel> kernels) {
  size_t global_work_size[1] = {size};
  cl_event e[2]; // todo dynamic
  // checkCudaErrors(cudaSetDevice(i));
  int j, k, ret;
  /* Major step */
  for (k = 2; k <= size; k <<= 1) {
    for (size_t i = 0; i < cq.size(); i++) {
      ret = clSetKernelArg(kernels[i], 2, sizeof(int), (void *)&k);
      assert(ret == CL_SUCCESS);
    }
    /* Minor step */
    for (j = k >> 1; j > 0; j = j >> 1) {
      for (size_t i = 0; i < cq.size(); i++) {
        ret = clSetKernelArg(kernels[i], 1, sizeof(int), (void *)&j);
        assert(ret == CL_SUCCESS);

        ret = clEnqueueNDRangeKernel(cq[i], kernels[i],
                                     1,                // work_dim
                                     0,                // global_work_offset
                                     global_work_size, // global_work_size
                                     NULL,             // local_work_size
                                     0,                // num_events_in_wait_list
                                     NULL,             // event_wait_list
                                     &e[i]             // event
                                     );
        assert(ret == CL_SUCCESS);
      }
      for (size_t i = 0; i < cq.size(); i++) {
        ret = clWaitForEvents(1, &e[i]);
        assert(ret == CL_SUCCESS);
        ret = clFinish(cq[i]);
        assert(ret == CL_SUCCESS);
      }
    }
  }
}

void Sort2::Start(unsigned int num_runs, const std::vector<int> options) {
  int ret = 0;
  int wg = 256;
  auto tid = this_thread::get_id();
  std::cout << DASH50 << "\n Sort Test\n";

  // decode options
  uint16_t power;
  if (options.size() > 0) {
    power = options[0];
  } else {
    cout << "Power of numbers to sort?: (0 for default)" << std::endl;
    power = promptValidated<int, int>("Power: ", [](int i) { return (i >= 0 && i <= 256); });
  }
  if (power == 0) {
    power = DEFAULTPOWER;
  }

  auto prog = cl::load_program("sort.cl", ctx, CtxDevices);

  /* Create Sapce for Random Numbers */
  cl_uint maxN = 1 << power;
  cl_uint maxNPC = (cl_uint)floor(maxN / cq.size());
  size_t sz = maxN * sizeof(cl_uint);
  size_t szPC = maxNPC * sizeof(cl_uint);
  cl_uint *rndData = new cl_uint[maxN];

  std::vector<cl_kernel> kernels;
  std::vector<cl_mem> inBuffers;
  for (size_t i = 0; i < cq.size(); i++) {
    /* Create OpenCL Kernel */
    kernels.push_back(clCreateKernel(prog, "bitonic_sort_step", &ret));
    assert(ret == CL_SUCCESS);
    // create buffers
    inBuffers.push_back(clCreateBuffer(ctx, CL_MEM_READ_WRITE, szPC, NULL, &ret));
    assert(ret == CL_SUCCESS);
    /* Set OpenCL Kernel Parameters */
    ret = clSetKernelArg(kernels[i], 0, sizeof(cl_mem), (void *)&inBuffers[i]);
    assert(ret == CL_SUCCESS);
  }

  unsigned int runs = 0;
  running = true;
  should_run = true;
  ResultFile r;
  r.name = "GpuCLParrallelSort2" + to_string(maxN);
  r.headdings = {"time_writebuffer"};

  while (ShouldRun() && runs < num_runs) {
    vector<unsigned long long> times;
    // printf(" thread Working\n");
    unsigned int percentDone = (unsigned int)(floor(((float)runs / (float)num_runs) * 100.0f));
    cout << "\r" << Spinner(runs) << "\t" << runs << "\tPercent Done: " << percentDone << "%"
         << std::flush;

    // make new numbers
    for (cl_uint i = 0; i < maxN; i++) {
      cl_uint x = (cl_uint)0;
      rndData[i] = (x << 14) | ((cl_uint)rand() & 0x3FFF);
      rndData[i] = (x << 14) | ((cl_uint)rand() & 0x3FFF);
    }
    Timer time_writebuffer;
    // send data
    for (size_t i = 0; i < cq.size(); i++) {
      cl_uint offset = (i * maxNPC);
      ret = clEnqueueWriteBuffer(cq[i], inBuffers[i], CL_TRUE, 0, szPC, &rndData[offset], 0, NULL,
                                 NULL); // blocking
      assert(ret == CL_SUCCESS);
      ret = clFinish(cq[i]); // Wait untill all commands executed.
      assert(ret == CL_SUCCESS);
    }
    for (auto q : cq) {
      ret = clFinish(q); // Wait untill all commands executed.
      assert(ret == CL_SUCCESS);
    }
    time_writebuffer.Stop();
    times.push_back(time_writebuffer.Duration_NS());
    /*
    * 2^numStages should be equal to length.
    * i.e the number of times you halve length to get 1 should be numStages
    */
    int temp;
    cl_int numStages = 0;
    for (temp = maxNPC; temp > 2; temp >>= 1)
      ++numStages;

    // run the sort.
    size_t nThreads[1];
    // nThreads[0] = (maxN / (2 * 4))/2;
    // nThreads[0] = maxNPC;
    nThreads[0] = maxNPC / (2 * 4);
   // cl_event e[2]; // todo dynamic
    unsigned int swapcount = 0;
    if (cq.size() == 2) {
      for (cl_uint swapsize = maxNPC / 2; swapsize > 0; swapsize /= 2) {
        if (runs == 0) {
          r.headdings.push_back("Sort_" + to_string(swapcount));
          r.headdings.push_back("Swap_" + to_string(swapsize));
        }
        Timer time_sort_inner;

        Sort(maxNPC, kernels);

        time_sort_inner.Stop();
        times.push_back(time_sort_inner.Duration_NS());
        Timer time_swap_inner;
        // now swap
        // read in the top of card 1
        cl_uint *tmpData = new cl_uint[swapsize];
        cl_uint a = swapsize * sizeof(cl_uint);
        ret = clEnqueueReadBuffer(cq[1], inBuffers[1], CL_TRUE, 0, a, tmpData, 0, NULL, NULL);
        assert(ret == CL_SUCCESS);

        ret = clFinish(cq[1]); // Wait untill all commands executed.
        assert(ret == CL_SUCCESS);

        // copy bottom of card 0 to top of card 1
        ret = clEnqueueCopyBuffer(cq[0], inBuffers[0], inBuffers[1],
                                  (maxNPC - swapsize) * sizeof(cl_uint), 0,
                                  swapsize * sizeof(cl_uint), 0, NULL, NULL);
        assert(ret == CL_SUCCESS);

        ret = clFinish(cq[0]);
        assert(ret == CL_SUCCESS);

        // write the top of card 1 to the bottom of card 0
        ret = clEnqueueWriteBuffer(cq[0], inBuffers[0], CL_TRUE,
                                   (maxNPC - swapsize) * sizeof(cl_uint),
                                   swapsize * sizeof(cl_uint), tmpData, 0, NULL, NULL);
        assert(ret == CL_SUCCESS);

        // Wait untill all commands executed.
        ret = clFinish(cq[0]);
        assert(ret == CL_SUCCESS);
        ret = clFinish(cq[1]);
        assert(ret == CL_SUCCESS);

        delete[] tmpData;
        time_swap_inner.Stop();
        times.push_back(time_swap_inner.Duration_NS());
        ++swapcount;
      }
    }
    Timer time_copyback;
    // Copy results from the memory buffer
    cl_uint *outData = new cl_uint[maxN];
    for (size_t i = 0; i < cq.size(); i++) {
      cl_uint offset = (i * maxNPC);
      ret = clEnqueueReadBuffer(cq[i], inBuffers[i], CL_TRUE, 0, szPC, &outData[offset], 0, NULL,
                                NULL);
      assert(ret == CL_SUCCESS);
      ret = clFinish(cq[i]); // Wait untill all commands executed.
      assert(ret == CL_SUCCESS);
    }
    time_copyback.Stop();
#if VERIFY
    assert(CheckArrayOrder(outData, maxN, true));
#endif
    delete outData;

    if (runs == 0) {
      r.headdings.push_back("Copy Back");
    }
    times.push_back(time_copyback.Duration_NS());
    r.times.push_back(times);
    ++runs;
  }

  delete[] rndData;
  r.CalcAvg();
  r.PrintToCSV(r.name);
  cout << "\n Sort finished\n";
  running = false;
};
