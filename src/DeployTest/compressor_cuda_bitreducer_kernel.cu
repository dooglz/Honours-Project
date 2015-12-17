#include <stdint.h>

__global__ void bit_reduce(const uint32_t *input_array, const uint16_t dataSize)
{
 unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

}


void Run_BitReduce(const dim3 a, const dim3 b, const uint32_t *input_array, const uint16_t dataSize) {
	bit_reduce <<<a, b >>> (input_array, dataSize);
	//cudaDeviceSynchronize();
}
