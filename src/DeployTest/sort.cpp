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
static  std::vector<cl_command_queue>  cq;
unsigned int Sort::GetMinCu() { return 1; }
unsigned int Sort::GetMax() { return 4; }
void Sort::Init(cl_context &context, std::vector<cl_command_queue>  &commandQ, std::vector<cl::Device> &devices,
                cl::Platform platform) {
  CtxDevices = devices;
  ctx = context;
  cq = commandQ;
}
void Sort::Shutdown() {}

void Sort::Work(unsigned int num_runs) {

  int wg = 256;
  auto tid = this_thread::get_id();
  std::cout << DASH50 << "\n Sort Test, Thread(" << tid << ")\n";
  char outstring[MEM_SIZE];

  auto prog = cl::load_program("sort.cl", ctx, CtxDevices);

  /* Create OpenCL Kernel */
  cl_int ret;
  auto kernel = clCreateKernel(prog, "bitonicSort2", &ret);
  assert(ret == CL_SUCCESS);

  /* Create Sapce for Random Numbers */
  cl_uint maxN = 1 << 16;
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
  //width
  ret = clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *)&maxN);
  assert(ret == CL_SUCCESS);
  //direction


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
    for (cl_uint i = 0; i < maxN; i++) {
      cl_uint x = (cl_uint)0;
      rndData[i] = (x << 14) | ((cl_uint)rand() & 0x3FFF);
      rndData[i] = (x << 14) | ((cl_uint)rand() & 0x3FFF);
    }
    // send data
    for(auto q: cq){
      ret = clEnqueueWriteBuffer(q, inBuffer, CL_TRUE, 0, sz, rndData, 0, NULL, NULL); // blocking
    }
    for (auto q : cq) {
      clFinish(q); // Wait untill all commands executed.
    }
   

    /*
    * 2^numStages should be equal to length.
    * i.e the number of times you halve length to get 1 should be numStages
    */
    int temp;
    cl_uint numStages = 0;
    for (temp = maxN; temp > 2; temp >>= 1)
      ++numStages;


    Timer t = Timer(to_string(runs));
    // run the sort.
    size_t nThreads[1];
    nThreads[0] = maxN /(2*4);
    size_t workGroup[1];
    workGroup[0] = wg;
    cl_event e;

    cl_int stage;
    cl_int passOfStage;

    for (stage = 0; stage < numStages; stage++) {
      // stage of the algorithm
      ret = clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *)&stage);
      assert(ret == CL_SUCCESS);
      // Every stage has stage + 1 passes
      for (passOfStage = stage; passOfStage >= 0; passOfStage--)
      {
        ret = clSetKernelArg(kernel, 2, sizeof(cl_uint), (void *)&passOfStage);
        assert(ret == CL_SUCCESS);

        /*
        * Enqueue a kernel run call.
        * For simplicity, the groupsize used is 1.
        *
        * Each thread writes a sorted pair.
        * So, the number of  threads (global) is half the length.
        */
        size_t global_work_size[1] = { passOfStage ? nThreads[0] : nThreads[0] << 1 };
        //ret = clEnqueueNDRangeKernel(cq, kernel, 1, 0, nThreads, workGroup, NULL, 0, &e);

        for (auto q : cq) {
          ret = clEnqueueNDRangeKernel(q, kernel, 1, 0, global_work_size, NULL, 0, NULL, &e);
          assert(ret == CL_SUCCESS);

        }
        for (auto q : cq) {
          clFinish(q); // Wait untill all commands executed.
        }
      }
    }

    //stop timer here
    t.Stop();
    
    // Copy results from the memory buffer
    for (auto q : cq) {
      cl_uint *outData = new cl_uint[maxN];
      ret = clEnqueueReadBuffer(q, inBuffer, CL_TRUE, 0, sz, outData, 0, NULL, NULL);
      clFinish(q); // Wait untill all commands executed.
    
//      assert(CheckArrayOrder(outData, maxN, false));
      delete outData;
    }
    ++runs;
    times.push_back(t);
  }
  delete rndData;
  PrintToCSV("Run #", "Time", times, "sort_" + current_time_and_date());
  printf("\n Sort finished\n");
  {

    std::lock_guard<std::mutex> lock(running_mutex);
    running = false;
  }
};
