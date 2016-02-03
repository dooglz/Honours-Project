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

Exp_Cuda_Compression::Exp_Cuda_Compression() : CudaExperiment(1, 4, "Cuda Compress", "Compresses Data") {}
Exp_Cuda_Compression::~Exp_Cuda_Compression() {}

static cl_context ctx;
static vector<cl::CLDevice> CtxDevices;
static std::vector<cl_command_queue> cq;
unsigned int Exp_Cuda_Compression::GetMinCu() { return 1; }
unsigned int Exp_Cuda_Compression::GetMax() { return 1; }
compressor_cuda_bitreducer *cmprsr;

void Exp_Cuda_Compression::Init(std::vector<cuda::CudaDevice> &devices) {
	//CtxDevices = devices;
	//ctx = context;
	//cq = commandQ;
	cmprsr = new compressor_cuda_bitreducer();
	cmprsr->Init();
}
void Exp_Cuda_Compression::Shutdown() {
	cmprsr->Shutdown();
	delete cmprsr;
	cmprsr = nullptr;
}

#define COUNT 32768
void Exp_Cuda_Compression::Start(unsigned int num_runs, const std::vector<int> options) {
	std::cout << "\n Cuda_Compress\n";
	//decode options
	bool uva = false;
	if (0){
		uva = cuda::enableUVA(0, 1);
		if (uva) {
			std::cout << "Using UVA P2P" << endl;
		}
	}
	const size_t dataSize = (COUNT)* sizeof(uint32_t);

	uint32_t* device_mem;
	cudaStream_t stream;
	uint32_t *host_mem;

	//malloc
	checkCudaErrors(cudaMallocHost((void**)&host_mem, dataSize));
	checkCudaErrors(cudaSetDevice(0)); //BIG FUCKING TODO HERE
	checkCudaErrors(cudaMalloc(&device_mem, dataSize));
	checkCudaErrors(cudaStreamCreate(&stream));

	//gen data
	for (size_t i = 0; i < COUNT; ++i) {
		uint32_t x = 0;
		host_mem[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
		host_mem[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
	}

	//send data to gpu
	checkCudaErrors(cudaMemcpyAsync(device_mem, host_mem, dataSize, cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaDeviceSynchronize());

	unsigned int runs = 0;
	running = true;
	should_run = true;

	ResultFile r;
	r.name = "Cuda_Compress_" + to_string(COUNT);
	r.headdings = cmprsr->TimingHeadders();

	vector<unsigned long long> times;
	cmprsr->EnableTiming(&times);

	while (ShouldRun() && runs < num_runs) {
		unsigned int percentDone = (unsigned int)(floor(((float)runs / (float)num_runs) * 100.0f));
		cout << "\r" << Spinner(runs) << "\t" << runs << "\tPercent Done: " << percentDone << "%"
			<< std::flush;

		times.clear();

		uint32_t* compressed_buffer;
		uint32_t compressed_size;
		cmprsr->Compress(device_mem, COUNT, compressed_buffer, compressed_size, stream);
		checkCudaErrors(cudaDeviceSynchronize());

		//cmprsr->Decompress();
		//checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFree(compressed_buffer));

		r.times.push_back(times);
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
