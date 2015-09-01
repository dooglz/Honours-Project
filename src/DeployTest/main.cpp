#include <iostream>
#include <vector>
#include <assert.h>
#include <CL/opencl.h>

#include "opencl_utils.h"
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

int print() {
  int i, j;
  char *value;
  size_t valueSize;
  cl_uint platformCount;
  cl_platform_id *platforms;
  cl_uint deviceCount;
  cl_uint maxComputeUnits;

  // get all platforms
  clGetPlatformIDs(0, NULL, &platformCount);
  platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platformCount);
  clGetPlatformIDs(platformCount, platforms, NULL);

  for (i = 0; i < platformCount; i++) {
    cl_device_id *devices;

	clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &valueSize);
	value = (char *)malloc(valueSize);
	clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, valueSize, value, NULL);
	printf("%d. Platform: %s\n", i + 1, value);
	free(value);

    // get all devices
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    devices = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceCount);
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
	
    // for each device print critical attributes
    for (j = 0; j < deviceCount; j++) {

      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
      value = (char *)malloc(valueSize);
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
      printf(" %d. Device: %s\n", j + 1, value);
      free(value);

      clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
      value = (char *)malloc(valueSize);
      clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
      printf("  %d.%d Hardware version: %s\n", j + 1, 1, value);
      free(value);

      clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
      value = (char *)malloc(valueSize);
      clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
      printf("  %d.%d Software version: %s\n", j + 1, 2, value);
      free(value);

      clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
      value = (char *)malloc(valueSize);
      clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
      printf("  %d.%d OpenCL C version: %s\n", j + 1, 3, value);
      free(value);

	  clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, 0, NULL, &valueSize);
	  value = (char *)malloc(valueSize);
	  clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, valueSize, value, NULL);
	  printf("  %d.%d OpenCL Device Type: %s\n", j + 1, 4, cl::DeviceTypeString((cl_device_type)value).c_str());
	  free(value);

	  clGetDeviceInfo(devices[j], CL_DEVICE_IMAGE_SUPPORT, 0, NULL, &valueSize);
	  value = (char *)malloc(valueSize);
	  clGetDeviceInfo(devices[j], CL_DEVICE_IMAGE_SUPPORT, valueSize, value, NULL);
	  printf("  %d.%d Image Support: %s\n", j + 1, 5, value ? "True" : "False");
	  free(value);

	  cl_ulong q = 0;
	  clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(q), &q, NULL);
	  printf("  %d.%d Global memory (bytes): %s\n", j + 1, 6, readable_fs(q).c_str());
	  clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(q), &q, NULL);
	  printf("  %d.%d Local memory (bytes): %s\n", j + 1, 7, readable_fs(q).c_str());

    cl_bool b = 0;
    clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(b), &b, NULL);
    printf("  %d.%d Unified memory: %s\n", j + 1, 8, b ? "True" : "False");

    cl_uint u = 0;
    clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(u), &u, NULL);
    printf("  %d.%d Clock frequency (MHz): %u\n", j + 1, 9, u );

    size_t t = 0;
    clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(t), &t, NULL);
    printf("  %d.%d max work-group size %u\n", j + 1, 10, t);

    clGetDeviceInfo(devices[j], CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(t), &t, NULL);
    printf("  %d.%d timer resolution (ns) %u\n", j + 1, 11, t);

      // print parallel compute units
      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits),
                      &maxComputeUnits, NULL);
      printf("  %d.%d Parallel compute units: %d\n", j + 12, 9, maxComputeUnits);
    }

    free(devices);
  }

  free(platforms);
  return 0;
}

int main() {
  std::cout << "Hello Deploy World!\n";
  print();
  cl::Init();
  std::vector<cl::device> devices;
  cl::GetRecommendedDevices(7, devices);
  std::cout << "\nRecommended devices:\n";
  for (auto dev : devices){
    std::cout << dev.short_name << "\n";
  }
  std::cout << "\nbye!\n";
}