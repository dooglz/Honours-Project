#include <stdint.h>

__global__ void bit_reduce(const uint32_t *input_array, const uint16_t dataSize, uint32_t *intBuf)
{
 uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
 uint32_t a = input_array[i];

 if (a <= 0xff) {
	 intBuf[i] = 1;
	 uint8_t b = static_cast<uint8_t>(a);
	 memcpy((uint8_t*)(&intBuf[i]) + 1, &b, sizeof(uint8_t));
 }
 else if (a <= 0xffff) {
	 intBuf[i] = sizeof(uint16_t);
	 uint16_t s = static_cast<uint16_t>(a);
	 memcpy((uint8_t*)(&intBuf[i]) + 1, &s, sizeof(uint16_t));
 }
 else {
	 intBuf[i] = sizeof(uint32_t);
	 memcpy((uint8_t*)(&intBuf[i]) + 1, &a, sizeof(uint32_t));
 }

}


void Run_BitReduce(const dim3 a, const dim3 b, const uint32_t *input_array, const uint16_t dataSize, uint32_t *intBuf) {
	bit_reduce << <a, b >> > (input_array, dataSize, intBuf);
	//cudaDeviceSynchronize();
}
