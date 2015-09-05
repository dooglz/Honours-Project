#include "opencl_utils.h"
#include <math.h>
#include <algorithm>
#include <functional>
#include <assert.h>
#include <iostream>
using namespace std;

namespace cl {
cl_uint total_num_devices;
cl_uint total_num_platforms;
vector<platform> platforms;
vector<device> devices;

const uint8_t Init() {
  platforms.clear();
  devices.clear();
  total_num_platforms = 0;
  total_num_devices = 0;

  // Status of OpenCL calls
  cl_int status;

  // Get the number of platforms
  status = clGetPlatformIDs(0, NULL, &total_num_platforms);
  assert(status == CL_SUCCESS);

  vector<cl_platform_id> platform_ids(total_num_platforms);

  status = clGetPlatformIDs(total_num_platforms, &platform_ids[0], nullptr);
  assert(status == CL_SUCCESS);
  std::vector<device> p_devices;
  for (auto id : platform_ids) {

    // Get platform devices
    cl_uint num_devices;
    status = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
    assert(status == CL_SUCCESS);

    vector<cl_device_id> devices_ids(num_devices);
    status = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, num_devices, &devices_ids[0], nullptr);
    assert(status == CL_SUCCESS);
    cl_uint total_cu = 0;

    for (auto dev_id : devices_ids) {
      // Get compute units and name info for each device
      cl_uint cu = 0;
      status = clGetDeviceInfo(dev_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &cu, NULL);
      assert(status == CL_SUCCESS);
      total_cu += cu;

	  //TODO tidy
	  p_devices.push_back(device{ dev_id, cu, id });
	  devices.push_back(device{ dev_id, cu, id });
	  clGetDeviceInfo(dev_id, CL_DEVICE_NAME, 32, &p_devices.back().short_name, NULL);
	  clGetDeviceInfo(dev_id, CL_DEVICE_NAME, 32, &devices.back().short_name, NULL);
    }
    total_num_devices += num_devices;

	platforms.push_back(platform{ id, total_cu, num_devices, "", p_devices});
    clGetPlatformInfo(id, CL_PLATFORM_NAME, 32, &platforms.back().short_name, NULL);
  }

  return 0;
}

// return an array of cl_device_id ordered by speed
const uint8_t GetFastestDevices(std::vector<device> &fastdevices) {
  if (total_num_devices < 1) {
    return 1;
  }
  fastdevices = devices;
  // get fastest devices, for now order by compliation units
  std::sort(fastdevices.begin(), fastdevices.end(),
            [](device const &a, device const &b) { return a.computeUnits < b.computeUnits; });
  return 0;
}

const uint8_t GetRecommendedDevices(const uint16_t count, std::vector<device> &devices) {
  if (total_num_devices < 1) {
    return 1;
  }
  if (total_num_devices < count) {
    return GetRecommendedDevices(total_num_devices, devices);
  }
  vector<device> fastdevices;
  GetFastestDevices(fastdevices);
  if (count == 1) {
    devices.push_back(fastdevices[0]);
    return 0;
  }

  // try to find a pair
  // split into platforms
  vector<cl_platform_id> suitable_p;
  uint16_t d_max = 0;
  for (auto p : platforms) {
    d_max = max(d_max, p.num_devices);
    if (p.num_devices >= count) {
      suitable_p.push_back(p.id);
    }
  }
  if (suitable_p.size() == 0) {
    // no platforms have count many Devices
    return GetRecommendedDevices(d_max, devices);
  } else if (suitable_p.size() == 1) {
    // just one platform that's suitable
    cl_platform_id p = suitable_p[0];
    cl_uint num_devices;
    clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
    if (num_devices > count) {
      // we have to choose who to leave out, go in order of speed
      for (auto d : fastdevices) {
        if (devices.size() == count) {
          break;
        }
        if (d.platform == p) {
          devices.push_back(d);
        }
      }
      return 0;
    } else if (num_devices == count) {
      // just return this platform
      for (auto d : fastdevices) {
        if (d.platform == p) {
          devices.push_back(d);
        }
      }
      return 0;
    }
  } else if (suitable_p.size() >= 1) {
    // now things get complicated
    // order the platforms by total speed
    // TODO: do all this in the init stage
    uint8_t *score = new uint8_t[suitable_p.size()];
    for (uint8_t i = 0; i < fastdevices.size(); i++) {
      device *d = &fastdevices[i];
      for (uint8_t j = 0; j < suitable_p.size(); j++) {
        if (d->platform == suitable_p[j]) {
          score[j] += fastdevices.size() - i;
        }
      }
    }
    // find the best scoring platform
    uint8_t winner = 0;
    for (uint8_t i = 1; i < suitable_p.size(); i++) {
      if (score[i] > score[winner]) {
        winner = i;
      }
    }
    delete[] score;

    cl_platform_id p = suitable_p[winner];
    cl_uint num_devices;
    clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
    if (num_devices > count) {
      // we have to choose who to leave out, go in order of speed
      for (auto d : fastdevices) {
        if (devices.size() == count) {
          break;
        }
        if (d.platform == p) {
          devices.push_back(d);
        }
      }
      return 0;
    } else if (num_devices == count) {
      // just return this platform
      for (auto d : fastdevices) {
        if (d.platform == p) {
          devices.push_back(d);
        }
      }
      return 0;
    }
  }

  return 0;
}

const std::string DeviceTypeString(const cl_device_type type) {
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

const void PrintInfo() {
  uint8_t i = 0;
  cl_uint u = 0;
  size_t t = 0;
  cl_bool b = 0;
  cl_ulong ul = 0;

  size_t valueSize;
  for (auto plat : platforms) {
    uint8_t j = 0;
    char *value;

    clGetPlatformInfo(plat.id, CL_PLATFORM_NAME, 0, NULL, &valueSize);
    value = (char *)malloc(valueSize);
    clGetPlatformInfo(plat.id, CL_PLATFORM_NAME, valueSize, value, NULL);
    cout << i + 1 << ".6 Platform: " << value << std::endl;
    free(value);

    for (auto dev : devices) {
      if (dev.platform != plat.id) {
        continue;
      }
      clGetDeviceInfo(dev.id, CL_DEVICE_NAME, 0, NULL, &valueSize);
      value = (char *)malloc(valueSize);
      clGetDeviceInfo(dev.id, CL_DEVICE_NAME, valueSize, value, NULL);
      printf(" %d. Device: %s\n", j + 1, value);
      free(value);

      clGetDeviceInfo(dev.id, CL_DEVICE_VERSION, 0, NULL, &valueSize);
      value = (char *)malloc(valueSize);
      clGetDeviceInfo(dev.id, CL_DEVICE_VERSION, valueSize, value, NULL);
      printf("  %d.%d Hardware version: %s\n", j + 1, 1, value);
      free(value);

      clGetDeviceInfo(dev.id, CL_DRIVER_VERSION, 0, NULL, &valueSize);
      value = (char *)malloc(valueSize);
      clGetDeviceInfo(dev.id, CL_DRIVER_VERSION, valueSize, value, NULL);
      printf("  %d.%d Software version: %s\n", j + 1, 2, value);
      free(value);

      clGetDeviceInfo(dev.id, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
      value = (char *)malloc(valueSize);
      clGetDeviceInfo(dev.id, CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
      printf("  %d.%d OpenCL C version: %s\n", j + 1, 3, value);
      free(value);

      clGetDeviceInfo(dev.id, CL_DEVICE_TYPE, 0, NULL, &valueSize);
      value = (char *)malloc(valueSize);
      clGetDeviceInfo(dev.id, CL_DEVICE_TYPE, valueSize, value, NULL);
      printf("  %d.%d OpenCL Device Type: %s\n", j + 1, 4,
             cl::DeviceTypeString((cl_device_type)value).c_str());
      free(value);

      clGetDeviceInfo(dev.id, CL_DEVICE_IMAGE_SUPPORT, sizeof(b), &b, NULL);
      printf("  %d.%d Image Support: %s\n", j + 1, 5, value ? "True" : "False");

      clGetDeviceInfo(dev.id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ul), &ul, NULL);
      cout << "  "<< j + 1 << ".6 Global memory (bytes):" << readable_fs(ul) << std::endl;

      clGetDeviceInfo(dev.id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ul), &ul, NULL);
      cout <<"  "<< j + 1 << ".7 Local memory (bytes):" << readable_fs(ul) << std::endl;

      clGetDeviceInfo(dev.id, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(b), &b, NULL);
      cout << "  " << j + 1 << ".8 Unified memory (bytes):" << (b ? "True" : "False") << std::endl;

      clGetDeviceInfo(dev.id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(u), &u, NULL);
      cout << "  " << j + 1 << ".9 clock frequency (MHz):" << u << std::endl;

      clGetDeviceInfo(dev.id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(t), &t, NULL);
      cout << "  " << j + 1 << ".10 max work-group size:" << t << std::endl;

      clGetDeviceInfo(dev.id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(t), &t, NULL);
      cout << "  " << j + 1 << ".11 timer resolution (ns):" << t << std::endl;

      clGetDeviceInfo(dev.id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(u), &u, NULL);
      cout << "  " << j + 1 << ".12 Parallel compute units:" << u << std::endl;
      ++j;
    }
    ++i;
  }
}
}