#include "exp_cuda_pingpong.h"
#include "utils.h"
#include "Timer.h"
#include <chrono>
#include <thread>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <string>

//#include <cuda.h>
#include <cuda_runtime.h>

Exp_Cuda_PingPong::Exp_Cuda_PingPong() : CudaExperiment(1, 4, "CudaPingPong", "Plays tennis with data") {}
Exp_Cuda_PingPong::~Exp_Cuda_PingPong() {}

static cl_context ctx;
static vector<cl::CLDevice> CtxDevices;
static std::vector<cl_command_queue> cq;
unsigned int Exp_Cuda_PingPong::GetMinCu() { return 2; }
unsigned int Exp_Cuda_PingPong::GetMax() { return 2; }

void Exp_Cuda_PingPong::Init(cl_context &context, std::vector<cl_command_queue> &commandQ,
	std::vector<cl::CLDevice> &devices, cl::Platform platform) {
	CtxDevices = devices;
	ctx = context;
	cq = commandQ;
}
void Exp_Cuda_PingPong::Shutdown() {}

#define COUNT 1024
void Exp_Cuda_PingPong::Start(unsigned int num_runs, const std::vector<int> options) {
	std::cout << "\n CudaPingPong\n";
	//decode options
	bool uva = false;
	if (0){
		uva = cuda::enableUVA(0, 1);
		if (uva) {
			std::cout << "Using UVA P2P" << endl;
		}
	}
	const size_t dataSize = (COUNT)* sizeof(uint32_t);

	uint32_t* device_mem[2];
	cudaStream_t streams[2];
	uint32_t *host_mem;

	//malloc
	checkCudaErrors(cudaMallocHost((void**)&host_mem, dataSize));
	for (size_t i = 0; i < 2; i++)
	{
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaMalloc(&device_mem[i], dataSize));
		checkCudaErrors(cudaStreamCreate(&streams[i]));
	}

	//enable p2p


	//gen data
	for (size_t i = 0; i < COUNT; ++i) {
		uint32_t x = 0;
		host_mem[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
		host_mem[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
	}
	//send data to gpu0
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMemcpyAsync(device_mem[0], host_mem, dataSize, cudaMemcpyHostToDevice, streams[0]));
	checkCudaErrors(cudaDeviceSynchronize());

	unsigned int runs = 0;
	running = true;
	should_run = true;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	ResultFile r;
	float time_ms;
	r.name = "Cuda_PingPong" + to_string(COUNT);
	r.headdings = { "A to B","B to A" };

	while (ShouldRun() && runs < num_runs) {
		vector<unsigned long long> times;

		//copy from 0 to 1
		checkCudaErrors(cudaEventRecord(start, streams[0]));
		if (uva) {
			checkCudaErrors(cudaMemcpyAsync(device_mem[1], device_mem[0], dataSize, cudaMemcpyDefault, streams[0]));
		}
		else{
			checkCudaErrors(cudaMemcpyPeerAsync(device_mem[1], 1, device_mem[0], 0, dataSize, streams[0]));
		}
		checkCudaErrors(cudaEventRecord(end, streams[0]));

		checkCudaErrors(cudaStreamSynchronize(streams[0]));
		checkCudaErrors(cudaEventElapsedTime(&time_ms, start, end));
		times.push_back(msFloatTimetoNS(time_ms));

		//copy back
		checkCudaErrors(cudaEventRecord(start, streams[0]));
		if (uva) {
			checkCudaErrors(cudaMemcpyAsync(device_mem[1], device_mem[0], dataSize, cudaMemcpyDefault, streams[0]));
		}
		else{
			checkCudaErrors(cudaMemcpyPeerAsync(device_mem[1], 1, device_mem[0], 0, dataSize, streams[0]));
		}
		checkCudaErrors(cudaEventRecord(end, streams[0]));

		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaStreamSynchronize(streams[0]));
		checkCudaErrors(cudaEventElapsedTime(&time_ms, start, end));

		times.push_back(msFloatTimetoNS(time_ms));
		r.times.push_back(times);
		++runs;
	}

	cudaEventDestroy(start);
	cudaEventDestroy(end);

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaFree(device_mem[0]));
	checkCudaErrors(cudaSetDevice(1));
	checkCudaErrors(cudaFree(device_mem[1]));
	checkCudaErrors(cudaFreeHost(host_mem));
	checkCudaErrors(cudaStreamDestroy(streams[0]));
	checkCudaErrors(cudaStreamDestroy(streams[1]));

	r.CalcAvg();
	r.PrintToCSV(r.name);
	cout << "\n Exp finished\n";
	running = false;
};
