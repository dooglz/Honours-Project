#include <stdint.h>
#include <string>
#include <vector>
#include <CL/opencl.h>

namespace ocl {
	struct device{
		cl_device_id id;
		uint16_t computeUnits;
		cl_platform_id platform;
	};

extern int global_variable; /* Declaration of the variable */
const uint8_t Init();
const std::string DeviceTypeString(const cl_device_type type);
const uint8_t GetRecommendedDevices(const uint8_t count, std::vector<device> &devices);
extern cl_uint num_devices;
}

const std::string readable_fs(const unsigned int size);