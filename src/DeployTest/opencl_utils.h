#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include <CL/opencl.h>

namespace cl {
extern cl_uint total_num_devices;
extern cl_uint total_num_platforms;

struct Platform;

struct CLDevice {
  cl_device_id id;
  cl_platform_id platform_id;
  Platform *platform;
  unsigned int computeUnits;
  std::string short_name;
};
struct Platform {
  cl_platform_id id;
  unsigned int computeUnits;
  unsigned int num_devices;
  std::string short_name;
  std::vector<CLDevice *> devices;
};

extern Platform *platforms;
extern CLDevice *CLdevices;

const unsigned int Init();
const std::string DeviceTypeString(const cl_device_type type);
const unsigned int GetRecommendedDevices(const unsigned int count,
                                         std::vector<CLDevice *> &devices);
const void PrintInfo();
const cl_int GetContext(const std::vector<CLDevice> &devices, cl_context &context,
                        std::vector<cl_command_queue> &cmd_queue);
cl_program load_program(const std::string &filename, cl_context &context,
                        const std::vector<CLDevice> &devices);

const void DeviceVectorToIdArray(const std::vector<CLDevice> &devices, cl_device_id *ids);
}

const std::string readable_fs(const unsigned int size);