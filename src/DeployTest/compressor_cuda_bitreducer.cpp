#pragma once
#include "compressor_cuda_bitreducer.h"
#include "cuda_utils.h"

#include <vector>
#include <chrono>

compressor_cuda_bitreducer::compressor_cuda_bitreducer(){}
compressor_cuda_bitreducer::~compressor_cuda_bitreducer(){}

void compressor_cuda_bitreducer::Init(){}
void compressor_cuda_bitreducer::Shutdown(){}

void Run_BitReduce(const dim3 a, const dim3 b, const uint32_t *input_array, const uint16_t dataSize, uint32_t *intBuf);

void compressor_cuda_bitreducer::Compress(const uint32_t* gpuBuffer, const size_t dataSize){

	//create output
	//TODO: have this passed in as param
	uint32_t* intBuf;
	const size_t int_buf_size = (dataSize* sizeof(uint32_t)) + (dataSize * sizeof(uint8_t));
	checkCudaErrors(cudaMalloc(&intBuf, int_buf_size));

	//Figure out our dimensions
	const unsigned int threadsperBlock = cuda::getBlockCount(512, dataSize);
	const dim3 blocks((dataSize / threadsperBlock), 1);
	const dim3 threads(threadsperBlock, 1);

	//reduce the bits
	Run_BitReduce(blocks, threads, gpuBuffer, (uint16_t)dataSize, intBuf);
	getLastCudaError("Run_BitReduce() execution failed.\n");

	checkCudaErrors(cudaDeviceSynchronize());
	//verify
	uint32_t* hostBuf;
	checkCudaErrors(cudaMallocHost((void**)&hostBuf, int_buf_size));
	checkCudaErrors(cudaMemcpy(hostBuf, intBuf, int_buf_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());

	//compact

	checkCudaErrors(cudaFree(intBuf));
	uint16_t qq = threadsperBlock;
}


void compressor_cuda_bitreducer::Decompress(){}
