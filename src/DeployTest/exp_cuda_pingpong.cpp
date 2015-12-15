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
unsigned int Exp_Cuda_PingPong::GetMinCu() { return 1; }
unsigned int Exp_Cuda_PingPong::GetMax() { return 2; }

void Exp_Cuda_PingPong::Init(cl_context &context, std::vector<cl_command_queue> &commandQ,
	std::vector<cl::CLDevice> &devices, cl::Platform platform) {
	CtxDevices = devices;
	ctx = context;
	cq = commandQ;
}
void Exp_Cuda_PingPong::Shutdown() {}

const int N = 16;
const int blocksize = 16;
const int threadsperblock = 512;

void Exp_Cuda_PingPong::Start(unsigned int num_runs, const std::vector<int> options) {
	std::cout << "\n CudaPingPong\n";

	unsigned int runs = 0;
	while (ShouldRun() && runs < num_runs) {

	}
	cout << "\n Exp finished\n";
	running = false;
};
