#pragma once
#include "compressor_cuda_bitreducer.h"
#include "cuda_utils.h"

#include <vector>
#include <chrono>

compressor_cuda_bitreducer::compressor_cuda_bitreducer(){}
compressor_cuda_bitreducer::~compressor_cuda_bitreducer(){}

void compressor_cuda_bitreducer::Init(){}
void compressor_cuda_bitreducer::Shutdown(){}

void Run_BitReduce(const dim3 a, const dim3 b, const uint32_t *input_array, const uint16_t dataSize);

void compressor_cuda_bitreducer::Compress(const uint32_t* gpuBuffer, const size_t dataSize){

	//create output buffer

	//Figure out our dimensions
	const unsigned int threadsperBlock = cuda::getBlockCount(512, dataSize);
	const dim3 blocks((dataSize / threadsperBlock), 1);
	const dim3 threads(threadsperBlock, 1);

	//reduce the bits
	Run_BitReduce(blocks, threads, gpuBuffer, (uint16_t)dataSize);
	getLastCudaError("Run_BitReduce() execution failed.\n");

	//compact

	uint16_t qq = threadsperBlock;
}


void compressor_cuda_bitreducer::Decompress(){}
