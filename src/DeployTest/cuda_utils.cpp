#include "cuda_utils.h"
#include "utils.h"
#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

using namespace std;

namespace cuda {

  const int getBlockCount(const int maxBlockSize, const  int threads){
  int a = maxBlockSize;
  while (a > 2 && (threads % a) != 0){
    a--;
  }
  return a;
}

int32_t total_num_devices;
CudaDevice *devices;

const unsigned int Init() {
  delete[] devices;
  total_num_devices = 0;

  cudaError_t error_id = cudaGetDeviceCount(&total_num_devices);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  return 0;
}

inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
  typedef struct {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {{0x20, 32},  // Fermi Generation (SM 2.0) GF100 class
                                     {0x21, 48},  // Fermi Generation (SM 2.1) GF10x class
                                     {0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
                                     {0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
                                     {0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
                                     {0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
                                     {0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
                                     {0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
                                     {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one to run properly
  printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor,
         nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

const void PrintInfo() {
  int dev, driverVersion = 0, runtimeVersion = 0;

  for (dev = 0; dev < total_num_devices; ++dev) {
    int j = 0;
#define idx dev << "." << j++ << "\t"

    cudaSetDevice(dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    cout << endl << DASH50 << endl;
    cout << "Device " << dev << ", " << p.name << endl;

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    cout << idx << "CUDA Driver Version:\t" << driverVersion / 1000 << "."
         << (driverVersion % 100) / 10 << endl;
    cout << idx << "CUDA Runtime Version:\t" << runtimeVersion / 1000 << "."
         << (runtimeVersion % 100) / 10 << endl;
    cout << idx << "CUDA Capability Major/Minor:\t" << p.major << "." << p.minor << endl;
    cout << idx << "Total amount of global memory:\t" << readable_fs(p.totalGlobalMem) << endl;
    cout << idx << "Multiprocessors:\t" << p.multiProcessorCount << endl;
    cout << idx << "CUDA Cores/MP:\t" << _ConvertSMVer2Cores(p.major, p.minor) << endl;
    cout << idx << "CUDA Cores:\t" << _ConvertSMVer2Cores(p.major, p.minor) * p.multiProcessorCount
         << endl;
    cout << idx << "GPU Max Clock rate:\t" << p.clockRate * 1e-3f << "MHz" << endl;

#if CUDART_VERSION >= 5000
    // This is supported in CUDA 5.0 (runtime API device properties)
    cout << idx << "Memory Clock rate:\t" << p.memoryClockRate * 1e-3f << "MHz" << endl;
    cout << idx << "Memory Bus Width:\t" << p.memoryBusWidth << "bit" << endl;

    if (p.l2CacheSize) {
      cout << idx << "L2 Cache Size:\t" << readable_fs(p.l2CacheSize) << endl;
    }
#endif
    cout << idx << "Maximum 1DTexture Dimension Size:\t" << p.maxTexture1D << endl;
    cout << idx << "Maximum 2DTexture Dimension Size:\t" << p.maxTexture2D[0] << ","
         << p.maxTexture2D[1] << endl;
    cout << idx << "Maximum 3DTexture Dimension Size:\t" << p.maxTexture3D[0] << ","
         << p.maxTexture3D[1] << "," << p.maxTexture3D[2] << endl;

    cout << idx << "Maximum Layered 1D Texture Size:\t" << p.maxTexture1DLayered[0] << ", "
         << p.maxTexture1DLayered[1] << " Layers" << endl;
    cout << idx << "Maximum Layered 2D Texture Size:\t" << p.maxTexture2DLayered[0] << ","
         << p.maxTexture2DLayered[1] << ", " << p.maxTexture1DLayered[1] << " Layers" << endl;

    cout << idx << "Total amount of constant memory:\t" << readable_fs(p.totalConstMem) << endl;
    cout << idx << "Total amount of shared memory per block:\t" << readable_fs(p.sharedMemPerBlock)
         << endl;
    cout << idx << "Total number of registers available per block:\t" << p.regsPerBlock << endl;
    cout << idx << "Warp size:\t" << p.warpSize << endl;
    cout << idx << "Maximum number of threads per multiprocessor:\t"
         << p.maxThreadsPerMultiProcessor << endl;
    cout << idx << "Maximum number of threads per block:\t" << p.maxThreadsPerBlock << endl;
    cout << idx << "Max dimension size of a thread block(x,y,z):\t" << p.maxThreadsDim[0] << ","
         << p.maxThreadsDim[1] << "," << p.maxThreadsDim[2] << endl;
    cout << idx << "Max dimension size of a thread block(x,y,z):\t" << p.maxThreadsDim[0] << ","
         << p.maxThreadsDim[1] << "," << p.maxThreadsDim[2] << endl;
    cout << idx << "Max dimension size of a grid size(x,y,z):\t" << p.maxGridSize[0] << ","
         << p.maxGridSize[1] << "," << p.maxGridSize[2] << endl;
    cout << idx << "Maximum memory pitch:\t" << readable_fs(p.memPitch) << endl;
    cout << idx << "Texture alignment:\t" << readable_fs(p.textureAlignment) << endl;

    cout << idx << "Concurrent copy and kernel execution:\t" << (p.deviceOverlap ? "Yes" : "No")
         << endl;
    cout << idx << "copy engine(s):\t" << p.asyncEngineCount << endl;

    cout << idx << "Run time limit on kernels:\t" << (p.kernelExecTimeoutEnabled ? "Yes" : "No")
         << endl;
    cout << idx << "Integrated GPU sharing Host Memory:\t" << (p.integrated ? "Yes" : "No") << endl;
    cout << idx << "Support host page-locked memory mapping:\t"
         << (p.canMapHostMemory ? "Yes" : "No") << endl;
    cout << idx << "Alignment requirement for Surfaces:\t" << (p.surfaceAlignment ? "Yes" : "No")
         << endl;
    cout << idx << "Device has ECC support:\t" << (p.ECCEnabled ? "Yes" : "No") << endl;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    cout << idx << "Device Driver Mode:\t" << (p.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                                                           : "WDDM (Windows Display Driver Model)")
         << endl;

#endif
    cout << idx << "Supports Unified Addressing (UVA):\t" << (p.unifiedAddressing ? "Yes" : "No")
         << endl;
    cout << idx << "PCI Domain ID / Bus ID / location ID:\t" << p.pciDomainID << "/" << p.pciBusID
         << "/" << p.pciDeviceID << endl;

    const char *sComputeMode[] = {
        "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
        "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this "
        "device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
        "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this "
        "device)",
        "Unknown", NULL};
    cout << idx << "ComputeMode:\t" << sComputeMode[p.computeMode] << endl;
  }

  // If there are 2 or more GPUs, query to determine whether RDMA is supported
  if (total_num_devices >= 2) {
    cudaDeviceProp prop[64];
    int gpuid[64]; // we want to find the first two GPUs that can support P2P
    int gpu_p2p_count = 0;

    for (int i = 0; i < total_num_devices; i++) {
      checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

      // Only boards based on Fermi or later can support P2P
      if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
          // on Windows (64-bit), the Tesla Compute Cluster driver for windows must be enabled to
          // support this
          && prop[i].tccDriver
#endif
          ) {
        // This is an array of P2P capable GPUs
        gpuid[gpu_p2p_count++] = i;
      }
    }

    // Show all the combinations of support P2P GPUs
    int can_access_peer;

    if (gpu_p2p_count >= 2) {
      for (int i = 0; i < gpu_p2p_count; i++) {
        for (int j = 0; j < gpu_p2p_count; j++) {
          if (gpuid[i] == gpuid[j]) {
            continue;
          }
          checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
          cout << ">Peer access from " << prop[gpuid[i]].name << " (" << gpuid[i] << ") ->"
               << prop[gpuid[j]].name << " (" << gpuid[j] << ") :\t"
               << (can_access_peer ? "Yes" : "No") << endl;
        }
      }
    }
  }

  // finish
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  cudaDeviceReset();
}

void cuda::__getLastCudaError(const char *errorMessage, const char *file, const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    cudaGetErrorString(err);
    cerr << " getLastCudaError() CUDA error, File:" << file << ", Line:" << line << endl
         << "Code:" << (int)err << "," << errorMessage << endl
         << cudaGetErrorString(err);

    /*
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line,
            errorMessage, (int)err, cudaGetErrorString(err));
    */
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

const char *cuda::_cudaGetErrorEnum(cudaError_t error) {
  switch (error) {
  case cudaSuccess:
    return "cudaSuccess";

  case cudaErrorMissingConfiguration:
    return "cudaErrorMissingConfiguration";

  case cudaErrorMemoryAllocation:
    return "cudaErrorMemoryAllocation";

  case cudaErrorInitializationError:
    return "cudaErrorInitializationError";

  case cudaErrorLaunchFailure:
    return "cudaErrorLaunchFailure";

  case cudaErrorPriorLaunchFailure:
    return "cudaErrorPriorLaunchFailure";

  case cudaErrorLaunchTimeout:
    return "cudaErrorLaunchTimeout";

  case cudaErrorLaunchOutOfResources:
    return "cudaErrorLaunchOutOfResources";

  case cudaErrorInvalidDeviceFunction:
    return "cudaErrorInvalidDeviceFunction";

  case cudaErrorInvalidConfiguration:
    return "cudaErrorInvalidConfiguration";

  case cudaErrorInvalidDevice:
    return "cudaErrorInvalidDevice";

  case cudaErrorInvalidValue:
    return "cudaErrorInvalidValue";

  case cudaErrorInvalidPitchValue:
    return "cudaErrorInvalidPitchValue";

  case cudaErrorInvalidSymbol:
    return "cudaErrorInvalidSymbol";

  case cudaErrorMapBufferObjectFailed:
    return "cudaErrorMapBufferObjectFailed";

  case cudaErrorUnmapBufferObjectFailed:
    return "cudaErrorUnmapBufferObjectFailed";

  case cudaErrorInvalidHostPointer:
    return "cudaErrorInvalidHostPointer";

  case cudaErrorInvalidDevicePointer:
    return "cudaErrorInvalidDevicePointer";

  case cudaErrorInvalidTexture:
    return "cudaErrorInvalidTexture";

  case cudaErrorInvalidTextureBinding:
    return "cudaErrorInvalidTextureBinding";

  case cudaErrorInvalidChannelDescriptor:
    return "cudaErrorInvalidChannelDescriptor";

  case cudaErrorInvalidMemcpyDirection:
    return "cudaErrorInvalidMemcpyDirection";

  case cudaErrorAddressOfConstant:
    return "cudaErrorAddressOfConstant";

  case cudaErrorTextureFetchFailed:
    return "cudaErrorTextureFetchFailed";

  case cudaErrorTextureNotBound:
    return "cudaErrorTextureNotBound";

  case cudaErrorSynchronizationError:
    return "cudaErrorSynchronizationError";

  case cudaErrorInvalidFilterSetting:
    return "cudaErrorInvalidFilterSetting";

  case cudaErrorInvalidNormSetting:
    return "cudaErrorInvalidNormSetting";

  case cudaErrorMixedDeviceExecution:
    return "cudaErrorMixedDeviceExecution";

  case cudaErrorCudartUnloading:
    return "cudaErrorCudartUnloading";

  case cudaErrorUnknown:
    return "cudaErrorUnknown";

  case cudaErrorNotYetImplemented:
    return "cudaErrorNotYetImplemented";

  case cudaErrorMemoryValueTooLarge:
    return "cudaErrorMemoryValueTooLarge";

  case cudaErrorInvalidResourceHandle:
    return "cudaErrorInvalidResourceHandle";

  case cudaErrorNotReady:
    return "cudaErrorNotReady";

  case cudaErrorInsufficientDriver:
    return "cudaErrorInsufficientDriver";

  case cudaErrorSetOnActiveProcess:
    return "cudaErrorSetOnActiveProcess";

  case cudaErrorInvalidSurface:
    return "cudaErrorInvalidSurface";

  case cudaErrorNoDevice:
    return "cudaErrorNoDevice";

  case cudaErrorECCUncorrectable:
    return "cudaErrorECCUncorrectable";

  case cudaErrorSharedObjectSymbolNotFound:
    return "cudaErrorSharedObjectSymbolNotFound";

  case cudaErrorSharedObjectInitFailed:
    return "cudaErrorSharedObjectInitFailed";

  case cudaErrorUnsupportedLimit:
    return "cudaErrorUnsupportedLimit";

  case cudaErrorDuplicateVariableName:
    return "cudaErrorDuplicateVariableName";

  case cudaErrorDuplicateTextureName:
    return "cudaErrorDuplicateTextureName";

  case cudaErrorDuplicateSurfaceName:
    return "cudaErrorDuplicateSurfaceName";

  case cudaErrorDevicesUnavailable:
    return "cudaErrorDevicesUnavailable";

  case cudaErrorInvalidKernelImage:
    return "cudaErrorInvalidKernelImage";

  case cudaErrorNoKernelImageForDevice:
    return "cudaErrorNoKernelImageForDevice";

  case cudaErrorIncompatibleDriverContext:
    return "cudaErrorIncompatibleDriverContext";

  case cudaErrorPeerAccessAlreadyEnabled:
    return "cudaErrorPeerAccessAlreadyEnabled";

  case cudaErrorPeerAccessNotEnabled:
    return "cudaErrorPeerAccessNotEnabled";

  case cudaErrorDeviceAlreadyInUse:
    return "cudaErrorDeviceAlreadyInUse";

  case cudaErrorProfilerDisabled:
    return "cudaErrorProfilerDisabled";

  case cudaErrorProfilerNotInitialized:
    return "cudaErrorProfilerNotInitialized";

  case cudaErrorProfilerAlreadyStarted:
    return "cudaErrorProfilerAlreadyStarted";

  case cudaErrorProfilerAlreadyStopped:
    return "cudaErrorProfilerAlreadyStopped";

  /* Since CUDA 4.0*/
  case cudaErrorAssert:
    return "cudaErrorAssert";

  case cudaErrorTooManyPeers:
    return "cudaErrorTooManyPeers";

  case cudaErrorHostMemoryAlreadyRegistered:
    return "cudaErrorHostMemoryAlreadyRegistered";

  case cudaErrorHostMemoryNotRegistered:
    return "cudaErrorHostMemoryNotRegistered";

  /* Since CUDA 5.0 */
  case cudaErrorOperatingSystem:
    return "cudaErrorOperatingSystem";

  case cudaErrorPeerAccessUnsupported:
    return "cudaErrorPeerAccessUnsupported";

  case cudaErrorLaunchMaxDepthExceeded:
    return "cudaErrorLaunchMaxDepthExceeded";

  case cudaErrorLaunchFileScopedTex:
    return "cudaErrorLaunchFileScopedTex";

  case cudaErrorLaunchFileScopedSurf:
    return "cudaErrorLaunchFileScopedSurf";

  case cudaErrorSyncDepthExceeded:
    return "cudaErrorSyncDepthExceeded";

  case cudaErrorLaunchPendingCountExceeded:
    return "cudaErrorLaunchPendingCountExceeded";

  case cudaErrorNotPermitted:
    return "cudaErrorNotPermitted";

  case cudaErrorNotSupported:
    return "cudaErrorNotSupported";

  /* Since CUDA 6.0 */
  case cudaErrorHardwareStackError:
    return "cudaErrorHardwareStackError";

  case cudaErrorIllegalInstruction:
    return "cudaErrorIllegalInstruction";

  case cudaErrorMisalignedAddress:
    return "cudaErrorMisalignedAddress";

  case cudaErrorInvalidAddressSpace:
    return "cudaErrorInvalidAddressSpace";

  case cudaErrorInvalidPc:
    return "cudaErrorInvalidPc";

  case cudaErrorIllegalAddress:
    return "cudaErrorIllegalAddress";

  /* Since CUDA 6.5*/
  case cudaErrorInvalidPtx:
    return "cudaErrorInvalidPtx";

  case cudaErrorInvalidGraphicsContext:
    return "cudaErrorInvalidGraphicsContext";

  case cudaErrorStartupFailure:
    return "cudaErrorStartupFailure";

  case cudaErrorApiFailureBase:
    return "cudaErrorApiFailureBase";
  }

  return "<unknown>";
}
}

// const cl_int GetContext(const std::vector<CLDevice> &devices, cl_context
// &context,std::vector<cl_command_queue> &cmd_queue);