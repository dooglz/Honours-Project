#include "opencl_utils.h"
#include <math.h>
#include <algorithm>
#include <functional>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

namespace cl {
cl_uint total_num_devices;
cl_uint total_num_platforms;
Platform *platforms;
Device *devices;

const unsigned int Init() {
  delete[] platforms;
  delete[] devices;
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

  // Quickly parse all devices
  for (auto id : platform_ids) {
    cl_uint num_devices;
    status = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
    assert(status == CL_SUCCESS);

    vector<cl_device_id> devices_ids(num_devices);
    status = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, num_devices, &devices_ids[0], nullptr);
    assert(status == CL_SUCCESS);
    total_num_devices += num_devices;
  }

  // setup storage
  platforms = new Platform[total_num_platforms];
  devices = new Device[total_num_devices];

  size_t deviceCount = 0;
  for (size_t i = 0; i < total_num_platforms; i++) {
    std::vector<Device *> p_devices;
    cl_platform_id id = platform_ids[i];
    cl_uint num_devices;
    status = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
    assert(status == CL_SUCCESS);

    vector<cl_device_id> devices_ids(num_devices);
    status = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, num_devices, &devices_ids[0], nullptr);
    assert(status == CL_SUCCESS);

    cl_uint total_cu = 0;
    size_t valueSize;
    clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, NULL, &valueSize);
    auto p_name = (char *)malloc(valueSize);
    clGetPlatformInfo(id, CL_PLATFORM_NAME, valueSize, p_name, NULL);
    platforms[i] = Platform{id, total_cu, num_devices, p_name};
    free(p_name);

    for (auto dev_id : devices_ids) {
      // Get compute units and name info for each device
      cl_uint cu = 0;
      status = clGetDeviceInfo(dev_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &cu, NULL);
      assert(status == CL_SUCCESS);
      total_cu += cu;

      // Deivce Name
      size_t valueSize;
      clGetDeviceInfo(dev_id, CL_DEVICE_NAME, 0, NULL, &valueSize);
      auto value = (char *)malloc(valueSize);
      clGetDeviceInfo(dev_id, CL_DEVICE_NAME, valueSize, value, NULL);
      string dev_name = value;
      free(value);

      devices[deviceCount] = Device{dev_id, id, &platforms[i], cu, dev_name};
      p_devices.push_back(&devices[deviceCount]);
      ++deviceCount;
    }
    platforms[i].devices = p_devices;
  }

  return 0;
}

// return an array of cl_device_id ordered by speed
const unsigned int GetFastestDevices(std::vector<Device *> &fastdevices) {
  if (total_num_devices < 1) {
    return 1;
  }
  for (size_t i = 0; i < total_num_devices; i++) {
    auto d = &devices[i];
    fastdevices.push_back(d);
  }
  // get fastest devices, for now order by compliation units
  std::sort(fastdevices.begin(), fastdevices.end(),
            [](Device *const &a, Device *const &b) { return a->computeUnits < b->computeUnits; });
  return 0;
}

const unsigned int GetRecommendedDevices(const unsigned int count, std::vector<Device *> &devices) {
  if (total_num_devices < 1) {
    return 1;
  }
  if (total_num_devices < count) {
    return GetRecommendedDevices(total_num_devices, devices);
  }
  vector<Device *> fastdevices;
  GetFastestDevices(fastdevices);
  if (count == 1) {
    devices.push_back(fastdevices[0]);
    return 0;
  }

  // try to find a pair
  // split into platforms
  vector<cl_platform_id> suitable_p;
  unsigned int d_max = 0;
  for (size_t i = 0; i < total_num_platforms; i++) {
    auto p = platforms[i];
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
        if (d->platform_id == p) {
          devices.push_back(d);
        }
      }
      return 0;
    } else if (num_devices == count) {
      // just return this platform
      for (auto d : fastdevices) {
        if (d->platform_id == p) {
          devices.push_back(d);
        }
      }
      return 0;
    }
  } else if (suitable_p.size() >= 1) {
    // now things get complicated
    // order the platforms by total speed
    // TODO: do all this in the init stage
    unsigned int *score = new unsigned int[suitable_p.size()];
    for (unsigned int i = 0; i < fastdevices.size(); i++) {
      Device *d = fastdevices[i];
      for (unsigned int j = 0; j < suitable_p.size(); j++) {
        if (d->platform_id == suitable_p[j]) {
          score[j] += fastdevices.size() - i;
        }
      }
    }
    // find the best scoring platform
    unsigned int winner = 0;
    for (unsigned int i = 1; i < suitable_p.size(); i++) {
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
        if (d->platform_id == p) {
          devices.push_back(d);
        }
      }
      return 0;
    } else if (num_devices == count) {
      // just return this platform
      for (auto d : fastdevices) {
        if (d->platform_id == p) {
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
  unsigned int i = 0;
  cl_uint u = 0;
  size_t t = 0;
  cl_bool b = 0;
  cl_ulong ul = 0;

  size_t valueSize;
  for (size_t i = 0; i < total_num_platforms; i++) {
    auto plat = platforms[i];
    unsigned int j = 0;
    char *value;

    clGetPlatformInfo(plat.id, CL_PLATFORM_NAME, 0, NULL, &valueSize);
    value = (char *)malloc(valueSize);
    clGetPlatformInfo(plat.id, CL_PLATFORM_NAME, valueSize, value, NULL);
    cout << i + 1 << ".6 Platform: " << value << std::endl;
    free(value);

    for (auto d : plat.devices) {
      auto dev = *d;
      if (dev.platform_id != plat.id) {
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
      cout << "  " << j + 1 << ".6 Global memory (bytes):" << readable_fs((unsigned int)ul)
           << std::endl;

      clGetDeviceInfo(dev.id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ul), &ul, NULL);
      cout << "  " << j + 1 << ".7 Local memory (bytes):" << readable_fs((unsigned int)ul)
           << std::endl;

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

const cl_int GetContext(const std::vector<Device> &devices, cl_context &context,
                        cl_command_queue &cmd_queue) {
  std::vector<cl_device_id> ids;
  for (auto d : devices) {
    ids.push_back(d.id);
  }
  cl_int status;
  // Create a context
  context = clCreateContext(nullptr, ids.size(), &ids[0], nullptr, nullptr, &status);
  assert(status == CL_SUCCESS);

  // Create a command queue
  cmd_queue = clCreateCommandQueue(context, ids[0], 0, &status);
  assert(status == CL_SUCCESS);
  return status;

  // cl_int clReleaseContext (	cl_context context)
}

cl_program load_program(const string &filename, cl_context &context, cl_device_id &device,
                        cl_int num_devices) {
  // Status of OpenCL calls
  cl_int status;

  // Create and compile program
  // Read in kernel file
  ifstream input(filename, ifstream::in);
  stringstream buffer;
  buffer << input.rdbuf();
  // Get the character array of the file contents
  auto file_contents = buffer.str();
  auto char_contents = file_contents.c_str();

  // Create program object
  auto program = clCreateProgramWithSource(context, 1, &char_contents, nullptr, &status);
  if (status != CL_SUCCESS) {
    cerr << "Error On clCreateProgramWithSource " << __LINE__ << " " << __FILE__ << endl;
    switch (status) {
    case CL_INVALID_CONTEXT:
      cerr << "CL_INVALID_CONTEXT" << endl;
      break;
    case CL_INVALID_VALUE:
      cerr << "CL_INVALID_VALUE" << endl;
      cerr << char_contents << endl;
      break;
    case CL_OUT_OF_HOST_MEMORY:
      cerr << "CL_OUT_OF_HOST_MEMORY" << endl;
      break;
    default:
      break;
    }
    assert(false);
  }

  // Compile / build program
  status = clBuildProgram(program, num_devices, &device, nullptr, nullptr, nullptr);

  // Check if compiled
  if (status != CL_SUCCESS) {
    // Error building - get log
    size_t length;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
    char *log = new char[length];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, log, &length);
    // Print log
    cout << log << endl;
    delete[] log;
  }

  // Return program object
  return program;
}
}