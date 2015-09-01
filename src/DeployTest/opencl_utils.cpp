#include "opencl_utils.h"
#include <math.h>

namespace ocl {
int global_variable = 66;

const void Init() {}

const std::string DeviceTypeString(cl_device_type type) {
  int ss = type;
  std::string s = "";
  if (type & CL_DEVICE_TYPE_CPU) {
    s += "CL_DEVICE_TYPE_CPU ";
  }
  if (type & CL_DEVICE_TYPE_GPU) {
    s += "CL_DEVICE_TYPE_GPU ";
  }
  if (type & CL_DEVICE_TYPE_ACCELERATOR) {
    s += "CL_DEVICE_TYPE_ACCELERATOR ";
  }
  if (type & CL_DEVICE_TYPE_DEFAULT) {
    s += "CL_DEVICE_TYPE_DEFAULT ";
  }
  return s;
}
}

std::string readable_fs(unsigned int sz /*in bytes*/) {
  float size = (float)sz;
  unsigned int kb = 1024;
  unsigned int mb = kb * 1024;
  unsigned int gb = mb * 1024;
  std::string s = "";
  float minus = 0;
  if (size > gb) {
    float a = floor(size / gb);
    minus += a * gb;
    s += std::to_string((int)a);
    s += "GB, ";
  }
  if (size > mb) {
    float a = floor((size - minus) / mb);
    minus += a * mb;
    s += std::to_string((int)a);
    s += "MB, ";
  }
  if (size > kb) {
    float a = floor((size - minus) / kb);
    minus += a * kb;
    s += std::to_string((int)a);
    s += "KB, ";
  }
  s += std::to_string((int)(size - minus));
  s += "B (";
  s += std::to_string(sz);
  s += ")";
  return s;
}