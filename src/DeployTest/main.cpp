#include <iostream>
#include <vector>
#include <assert.h>
#include <CL/opencl.h>
//
#include "opencl_utils.h"
#include "sort.h"
// for sleep
#include <chrono>
#include <thread>

using namespace std;

// Initialise OpenCL
void initialise_opencl(vector<cl_platform_id> &platforms, vector<cl_device_id> &devices,
                       cl_context &context, cl_command_queue &cmd_queue) {
  // Status of OpenCL calls
  cl_int status;

  // Get the number of platforms
  cl_uint num_platforms;
  status = clGetPlatformIDs(0, nullptr, &num_platforms);
  assert(status != CL_SUCCESS);

  // Resize vector to store platforms
  platforms.resize(num_platforms);

  // Fill in platform vector
  status = clGetPlatformIDs(num_platforms, &platforms[0], nullptr);
  assert(status != CL_SUCCESS);

  // Assume platform 0 is the one we want to use
  // Get devices for platform 0
  cl_uint num_devices;
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  assert(status != CL_SUCCESS);

  // Resize vector to store devices
  devices.resize(num_devices);
  // Fill in devices vector
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, &devices[0], nullptr);
  assert(status != CL_SUCCESS);

  // Create a context
  context = clCreateContext(nullptr, num_devices, &devices[0], nullptr, nullptr, &status);

  // Create a command queue
  cmd_queue = clCreateCommandQueue(context, devices[0], 0, &status);
}
int main() {
  std::cout << "Hello Deploy World!\n";
  cl::Init();
  cl::PrintInfo();
  std::vector<cl::device> devices;
  cl::GetRecommendedDevices(7, devices);
  std::cout << "\nRecommended devices:\n";
  for (auto dev : devices) {
    std::cout << dev.short_name << "\n";
  }
  Sort *srt = new Sort();
  srt->Start(1501);
  //  std::this_thread::sleep_for(std::chrono::seconds(6));
  while (srt->IsRunning()) {
  }
  delete srt;
  std::cout << "\nbye!\n";
}