#include <stdint.h>
#include <string>
#include <vector>
#include <CL/opencl.h>

namespace cl {
  extern cl_uint total_num_devices;
  extern cl_uint total_num_platforms;
	struct device{
		cl_device_id id;
		uint16_t computeUnits;
		cl_platform_id platform;
    char short_name [32];
	};
  struct platform{
    cl_platform_id id;
    uint16_t computeUnits;
    uint16_t num_devices;
    char short_name[32];
  };
const uint8_t Init();
const std::string DeviceTypeString(const cl_device_type type);
const uint8_t GetRecommendedDevices(const uint8_t count, std::vector<device> &devices);

}

const std::string readable_fs(const unsigned int size);