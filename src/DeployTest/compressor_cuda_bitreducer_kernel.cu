#include <stdint.h>

__global__ void bit_reduce(const uint32_t *input_array, uint32_t *intBuf)
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

__global__ void seq_compact(uint8_t *intBuf, const uint16_t dataSize, uint32_t* sizeBuf)
{
	uint16_t readIndex = 0;
	uint16_t writeIndex = 0;
	for (uint16_t i = 0; i < dataSize; ++i)
	{
		const uint16_t readIndex = i * 4;
		uint8_t size = intBuf[readIndex];
		memcpy(&intBuf[writeIndex], &intBuf[readIndex], size +1);
		writeIndex += size + 1;
	}
	sizeBuf[0] = writeIndex;

	//zero out the rest of the buffer
	const uint32_t int_buf_size = (dataSize* sizeof(uint32_t)) + (dataSize * sizeof(uint8_t));
	memset(&intBuf[writeIndex], 0, int_buf_size - int_buf_size);
}


void Run_BitReduce(const dim3 a, const dim3 b, const uint32_t *input_array, uint32_t *intBuf) {
	bit_reduce << <a, b >> > (input_array, intBuf);
}
void Run_seq_compact(const uint32_t *input_array, const uint16_t dataSize, uint32_t* sizeBuf) {
	const dim3 blocks(1, 1);
	seq_compact << <blocks, blocks >> > ((uint8_t *)input_array, dataSize, sizeBuf);
}
