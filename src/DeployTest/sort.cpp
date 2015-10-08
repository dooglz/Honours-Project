#include "sort.h"
#include "utils.h"
#include <chrono> // std::chrono::seconds
#include <thread>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#define MEM_SIZE (128)
#define VERIFY true
Sort::Sort() : Experiment(1, 4, "Sort", "Sorts Things") {}

Sort::~Sort() {}

static cl_context ctx;
static vector<cl::Device> CtxDevices;
static std::vector<cl_command_queue> cq;
unsigned int Sort::GetMinCu() { return 1; }
unsigned int Sort::GetMax() { return 4; }
void Sort::Init(cl_context &context, std::vector<cl_command_queue> &commandQ,
                std::vector<cl::Device> &devices, cl::Platform platform) {
  CtxDevices = devices;
  ctx = context;
  cq = commandQ;
}
void Sort::Shutdown() {}

void Sort::Work(unsigned int num_runs) {
  int ret = 0;
  int wg = 256;
  auto tid = this_thread::get_id();
  std::cout << DASH50 << "\n Sort Test, Thread(" << tid << ")\n";

  auto prog = cl::load_program("sort.cl", ctx, CtxDevices);

  /* Create Sapce for Random Numbers */
  cl_uint maxN = 1 << 8;
  cl_uint maxNPC = (cl_uint)floor(maxN / cq.size());
  size_t sz = maxN * sizeof(cl_uint);
  size_t szPC = maxNPC * sizeof(cl_uint);
  cl_uint *rndData = new cl_uint[maxN];

  std::vector<cl_kernel> kernels;
  std::vector<cl_mem> inBuffers;
  for (size_t i = 0; i < cq.size(); i++) {
    /* Create OpenCL Kernel */
    kernels.push_back(clCreateKernel(prog, "bitonicSort2", &ret));
    assert(ret == CL_SUCCESS);
    // create buffers
    inBuffers.push_back(clCreateBuffer(ctx, CL_MEM_READ_ONLY, szPC, NULL, &ret));
    assert(ret == CL_SUCCESS);
    /* Set OpenCL Kernel Parameters */
    ret = clSetKernelArg(kernels[i], 0, sizeof(cl_mem), (void *)&inBuffers[i]);
    assert(ret == CL_SUCCESS);
    // width
    ret = clSetKernelArg(kernels[i], 3, sizeof(cl_uint), (void *)&maxNPC);
    assert(ret == CL_SUCCESS);
  }

  unsigned int runs = 0;
  {
    std::lock_guard<std::mutex> lock(running_mutex);
    running = true;
  }

  std::vector<Timer> times;
  while (ShouldRun() && runs < num_runs) {

    // printf(" thread Working\n");
    unsigned int percentDone = (unsigned int)(floor(((float)runs / (float)num_runs) * 100.0f));
    wprintf(L" %c\t%u\tPercent Done: %u%%  \t\t\r", Spinner(runs), runs, percentDone);
    //

    // make new numbers
    for (cl_uint i = 0; i < maxN; i++) {
      cl_uint x = (cl_uint)0;
      rndData[i] = (x << 14) | ((cl_uint)rand() & 0x3FFF);
      rndData[i] = (x << 14) | ((cl_uint)rand() & 0x3FFF);
    }
    // send data
    for (size_t i = 0; i < cq.size(); i++) {
      cl_uint offset = (i * maxNPC);
      ret = clEnqueueWriteBuffer(cq[i], inBuffers[i], CL_TRUE, 0, szPC, &rndData[offset], 0, NULL,
                                 NULL); // blocking
      assert(ret == CL_SUCCESS);
    }
    for (auto q : cq) {
      clFinish(q); // Wait untill all commands executed.
    }

    /*
    * 2^numStages should be equal to length.
    * i.e the number of times you halve length to get 1 should be numStages
    */
    int temp;
    cl_int numStages = 0;
    for (temp = maxN; temp > 2; temp >>= 1)
      ++numStages;

    Timer t = Timer(to_string(runs));
    // run the sort.
    size_t nThreads[1];
    nThreads[0] = maxN / (2 * 4);
    cl_event e;

    if (cq.size() == 2) {
      for (cl_uint swapsize = maxNPC / 2; swapsize > 0; swapsize /= 2) {
        for (cl_int stage = 0; stage < numStages; stage++) {
          // stage of the algorithm
          for (size_t i = 0; i < cq.size(); i++) {
            ret = clSetKernelArg(kernels[i], 1, sizeof(cl_uint), (void *)&stage);
            assert(ret == CL_SUCCESS);
          }
          // Every stage has stage + 1 passes
          for (cl_int passOfStage = stage; passOfStage >= 0; passOfStage--) {
            for (size_t i = 0; i < cq.size(); i++) {
              ret = clSetKernelArg(kernels[i], 2, sizeof(cl_uint), (void *)&passOfStage);
              assert(ret == CL_SUCCESS);
            }

            size_t global_work_size[1] = {passOfStage ? nThreads[0] : nThreads[0] << 1};

            for (size_t i = 0; i < cq.size(); i++) {
              ret = clEnqueueNDRangeKernel(cq[i], kernels[i], 1, 0, global_work_size, NULL, 0, NULL,
                                           &e);
              assert(ret == CL_SUCCESS);
            }
            for (auto q : cq) {
              clFinish(q); // Wait untill all commands executed.
            }
          }
        }

        // now swap
        // read in the top of card 1
        cl_uint *tmpData = new cl_uint[swapsize];
        ret = clEnqueueReadBuffer(cq[1], inBuffers[1], CL_TRUE, 0, swapsize * sizeof(cl_uint),
                                  tmpData, 0, NULL, NULL);
        clFinish(cq[1]); // Wait untill all commands executed.
        // copy bottom of card 0 to top of card 1
        clEnqueueCopyBuffer(cq[0], inBuffers[0], inBuffers[1],
                            (maxNPC - swapsize) * sizeof(cl_uint), 0, swapsize * sizeof(cl_uint), 0,
                            NULL, NULL);
        clFinish(cq[0]); // Wait untill all commands executed.
                         // write the top of card 1 to the bottom of card 0
        clEnqueueWriteBuffer(cq[0], inBuffers[0], CL_TRUE, (maxNPC - swapsize) * sizeof(cl_uint),
                             swapsize * sizeof(cl_uint), tmpData, 0, NULL, NULL);
        delete[] tmpData;
      }
    }

    // do one final sort, or possibly the first sort if 1 gpu
    for (cl_int stage = 0; stage < numStages; stage++) {
      // stage of the algorithm
      for (size_t i = 0; i < cq.size(); i++) {
        ret = clSetKernelArg(kernels[i], 1, sizeof(cl_uint), (void *)&stage);
        assert(ret == CL_SUCCESS);
      }
      // Every stage has stage + 1 passes
      for (cl_int passOfStage = stage; passOfStage >= 0; passOfStage--) {
        for (size_t i = 0; i < cq.size(); i++) {
          ret = clSetKernelArg(kernels[i], 2, sizeof(cl_uint), (void *)&passOfStage);
          assert(ret == CL_SUCCESS);
        }

        size_t global_work_size[1] = {passOfStage ? nThreads[0] : nThreads[0] << 1};

        for (size_t i = 0; i < cq.size(); i++) {
          ret =
              clEnqueueNDRangeKernel(cq[i], kernels[i], 1, 0, global_work_size, NULL, 0, NULL, &e);
          assert(ret == CL_SUCCESS);
        }
        for (auto q : cq) {
          clFinish(q); // Wait untill all commands executed.
        }
      }
    }

    // stop timer here
    t.Stop();
#if VERIFY
    // Copy results from the memory buffer
    cl_uint *outData = new cl_uint[maxN];
    for (size_t i = 0; i < cq.size(); i++) {
      cl_uint offset = (i * maxNPC);
      ret = clEnqueueReadBuffer(cq[i], inBuffers[i], CL_TRUE, 0, szPC, &outData[offset], 0, NULL,
                                NULL);
      clFinish(cq[i]); // Wait untill all commands executed.
    }
    assert(CheckArrayOrder(outData, maxN, false));
    delete outData;
#endif
    ++runs;
    times.push_back(t);
  }
  delete[] rndData;
  PrintToCSV("Run #", "Time", times, "sort_" + current_time_and_date());
  printf("\n Sort finished\n");
  {

    std::lock_guard<std::mutex> lock(running_mutex);
    running = false;
  }
};
