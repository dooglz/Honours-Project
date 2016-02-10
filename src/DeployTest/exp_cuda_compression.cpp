#include "exp_cuda_compression.h"
#include "compressor_cuda_bitreducer.h"

#include "utils.h"
#include "timer.h"
#include <chrono>
#include <thread>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <string>

//#include <cuda.h>
#include <cuda_runtime.h>

Exp_Cuda_Compression::Exp_Cuda_Compression() : CudaExperiment(1, 4, "Cuda_Bit_Compressor", "Compresses Data") {}
Exp_Cuda_Compression::~Exp_Cuda_Compression() {}


static vector<cuda::CudaDevice> CtxDevices;
unsigned int Exp_Cuda_Compression::GetMinCu() { return 1; }
unsigned int Exp_Cuda_Compression::GetMax() { return 1; }
compressor_cuda_bitreducer *cmprsr;

void Exp_Cuda_Compression::Init(std::vector<cuda::CudaDevice> &devices) {
	CtxDevices = devices;
	cmprsr = new compressor_cuda_bitreducer();
	cmprsr->Init();
}
void Exp_Cuda_Compression::Shutdown() {
	cmprsr->Shutdown();
	delete cmprsr;
	cmprsr = nullptr;
}

#define DEFAULTPOWER 18
void Exp_Cuda_Compression::Start(unsigned int num_runs, const std::vector<int> options) {
  if (CtxDevices.size() < GetMinCu() || CtxDevices.size() > GetMax()) {
    std::cout << "\n invalid number of devices\n";
    return;
  }
	std::cout << "\n Cuda_Bit_Compress\n";
  // decode options
  uint16_t power;
  if (options.size() > 0) {
    power = options[0];
  }
  else {
    cout << "Power of numbers to swap?: (0 for default)" << std::endl;
    power = promptValidated<int, int>("Power: ", [](int i) { return (i >= 0 && i <= 256); });
  }
  if (power == 0) {
    power = DEFAULTPOWER;
  }

  const uint32_t count = 1 << power;
  const size_t dataSize = (count)* sizeof(uint32_t);

	uint32_t* device_mem;
	cudaStream_t stream;
	uint32_t *host_mem;

	//malloc
	checkCudaErrors(cudaMallocHost((void**)&host_mem, dataSize));
  checkCudaErrors(cudaSetDevice(CtxDevices[0].id));
	checkCudaErrors(cudaMalloc(&device_mem, dataSize));
	checkCudaErrors(cudaStreamCreate(&stream));

	//gen data
  for (size_t i = 0; i < count; ++i) {
		uint32_t x = 0;
		host_mem[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
		host_mem[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
	}

	//send data to gpu
  cudaEvent_t transfer_start, transfer_end;
  //cudaEventCreate(&transfer_start);
  //cudaEventCreate(&transfer_end);

//  checkCudaErrors(cudaEventRecord(transfer_start, stream));
	checkCudaErrors(cudaMemcpyAsync(device_mem, host_mem, dataSize, cudaMemcpyHostToDevice, stream));
  //checkCudaErrors(cudaEventRecord(transfer_end, stream));
	checkCudaErrors(cudaDeviceSynchronize());

	unsigned int runs = 0;
	running = true;
	should_run = true;

	ResultFile r;
  r.name = "Cuda_Bit_Compressor_" + to_string(count);
	r.headdings = cmprsr->TimingHeadders();
  //r.headdings = { r.name, "Ratio" };

	vector<unsigned long long> times;
	cmprsr->EnableTiming(&times);

	while (ShouldRun() && runs < num_runs) {
		unsigned int percentDone = (unsigned int)(floor(((float)runs / (float)num_runs) * 100.0f));
		cout << "\r" << Spinner(runs) << "\t" << runs << "\tPercent Done: " << percentDone << "%"
			<< std::flush;
    
//regen
    for (size_t i = 0; i < count; ++i) {
      uint32_t x = 0;
      host_mem[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
      host_mem[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
    }
    checkCudaErrors(cudaMemcpyAsync(device_mem, host_mem, dataSize, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaDeviceSynchronize());

		times.clear();

		uint32_t* compressed_buffer;
		uint32_t compressed_size;

    cudaEvent_t compress_start, compress_end;
    //cudaEventCreate(&compress_start);
   // cudaEventCreate(&compress_end);

   // checkCudaErrors(cudaEventRecord(compress_start, stream));
    cmprsr->Compress(device_mem, count, compressed_buffer, compressed_size, stream);
    //checkCudaErrors(cudaEventRecord(compress_end, stream));
		checkCudaErrors(cudaDeviceSynchronize());


		//cmprsr->Decompress();
		//checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFree(compressed_buffer));

    float time_ms = 0.0f;
   // checkCudaErrors(cudaEventElapsedTime(&time_ms, compress_start, compress_end));
  //  const float ratio = (float)compressed_size / (float)(dataSize * sizeof(uint32_t));
   // r.times.push_back({ msFloatTimetoNS(time_ms), (unsigned long long)(ratio*100.0f) });

		r.times.push_back(times);

   // cudaEventDestroy(compress_start);
   // cudaEventDestroy(compress_end);
   // cudaEventDestroy(transfer_start);
    //cudaEventDestroy(transfer_end);

		++runs;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(device_mem));
	//checkCudaErrors(cudaFreeHost((void*)host_mem));

	checkCudaErrors(cudaStreamDestroy(stream));

	checkCudaErrors(cudaDeviceReset());

	r.CalcAvg();
	r.PrintToCSV(r.name);
	std::cout << "\n Exp finished\n";
	running = false;
};
