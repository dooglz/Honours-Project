#include <string>
#include <CL/opencl.h>

namespace ocl {
extern int global_variable; /* Declaration of the variable */
const void Init();
const std::string DeviceTypeString(cl_device_type type);
}

std::string readable_fs(unsigned int size);