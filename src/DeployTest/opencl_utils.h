#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include <CL/opencl.h>

namespace cl {
extern cl_uint total_num_devices;
extern cl_uint total_num_platforms;

struct device {
  cl_device_id id;
  cl_platform_id platform;
  unsigned int computeUnits;
  char short_name[32];
};
struct platform {
  cl_platform_id id;
  unsigned int computeUnits;
  unsigned int num_devices;
  char short_name[32];
  std::vector<device> devices;
};

extern std::vector<platform> platforms;
extern std::vector<device> devices;

const unsigned int Init();
const std::string DeviceTypeString(const cl_device_type type);
const unsigned int GetRecommendedDevices(const unsigned int count, std::vector<device> &devices);
const void PrintInfo();
const cl_int GetContext(const std::vector<device> &devices, cl_context &context,
                        cl_command_queue &cmd_queue);
}

const std::string readable_fs(const unsigned int size);