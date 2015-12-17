#pragma once
#include "compressor_cuda_bitreducer.h"
#include "cuda_utils.h"

#include <vector>
#include <chrono>

compressor_cuda_bitreducer::compressor_cuda_bitreducer(){}
compressor_cuda_bitreducer::~compressor_cuda_bitreducer(){}

void compressor_cuda_bitreducer::Init(){}
void compressor_cuda_bitreducer::Shutdown(){}

void Run_BitReduce(const dim3 a, const dim3 b, const uint32_t *input_array, uint32_t *intBuf);
void Run_seq_compact(const uint32_t *input_array, const uint16_t dataSize, uint32_t* sizeBuf);

void compressor_cuda_bitreducer::Compress(const uint32_t* gpuBuffer, const size_t dataSize, uint32_t* &outBuffer, uint32_t &outSize){

	//create output
	const size_t int_buf_size = (dataSize* sizeof(uint32_t)) + (dataSize * sizeof(uint8_t));
	checkCudaErrors(cudaMalloc(&outBuffer, int_buf_size));

	//Figure out our dimensions
	const unsigned int threadsperBlock = cuda::getBlockCount(512, dataSize);
	const dim3 blocks((dataSize / threadsperBlock), 1);
	const dim3 threads(threadsperBlock, 1);

	//reduce the bits
	Run_BitReduce(blocks, threads, gpuBuffer, outBuffer);
	getLastCudaError("Run_BitReduce() execution failed.\n");
	checkCudaErrors(cudaDeviceSynchronize());

	//compact
	uint32_t* sizeBuf;
	checkCudaErrors(cudaMalloc(&sizeBuf, 4));

	Run_seq_compact(outBuffer, dataSize, sizeBuf);
	getLastCudaError("Run_seq_compact() execution failed.\n");
	checkCudaErrors(cudaMemcpy(&outSize, sizeBuf, 4, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(sizeBuf));
	checkCudaErrors(cudaDeviceSynchronize());

	const float ratio = (float)outSize / (float)(dataSize* sizeof(uint32_t));
}


void compressor_cuda_bitreducer::Decompress(){}
