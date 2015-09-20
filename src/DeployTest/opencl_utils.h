#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include <CL/opencl.h>

namespace cl {
extern cl_uint total_num_devices;
extern cl_uint total_num_platforms;

struct platform;
struct device;

struct device {
  cl_device_id id;
  cl_platform_id platform_id;
  platform *platform;
  unsigned int computeUnits;
  std::string short_name;
};
struct platform {
  cl_platform_id id;
  unsigned int computeUnits;
  unsigned int num_devices;
  std::string short_name;
  std::vector<device *> devices;
};

extern platform *platforms;
extern device *devices;

const unsigned int Init();
const std::string DeviceTypeString(const cl_device_type type);
const unsigned int GetRecommendedDevices(const unsigned int count, std::vector<device*> &devices);
const void PrintInfo();
const cl_int GetContext(const std::vector<device> &devices, cl_context &context,
                        cl_command_queue &cmd_queue);
cl_program load_program(const std::string &filename, cl_context &context, cl_device_id &device,
                        cl_int num_devices);
}

const std::string readable_fs(const unsigned int size);