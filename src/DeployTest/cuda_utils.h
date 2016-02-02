#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace cuda {
extern int32_t total_num_devices;

struct CudaDevice {
  unsigned int id;
  // cl_platform_id platform_id;
  unsigned int computeUnits;
  std::string short_name;
};

extern CudaDevice *CudaDevices;

const unsigned int Init();
// const std::string DeviceTypeString(const cl_device_type type);
// const unsigned int GetRecommendedDevices(const unsigned int count, std::vector<CLDevice *>
// &devices);
const void PrintInfo();
/*
const cl_int GetContext(const std::vector<CLDevice> &devices, cl_context &context,
                        std::vector<cl_command_queue> &cmd_queue);
                        */
// const void DeviceVectorToIdArray(const std::vector<CLDevice> &devices, cl_device_id *ids);
const char *_cudaGetErrorEnum(cudaError_t error);
void __getLastCudaError(const char *errorMessage, const char *file, const int line);

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    cudaDeviceReset();
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}
#define checkCudaErrors(val) cuda::check((val), #val, __FILE__, __LINE__)
// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) cuda::__getLastCudaError(msg, __FILE__, __LINE__)

const int getBlockCount(const int maxBlockSize, const int threads);

const bool enableUVA(const int gpu0, const int gpu1);
const bool enableP2P(const int gpu0, const int gpu1);
}
