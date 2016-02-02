#include <stdint.h>
#include <stdio.h>
__global__ void bit_reduce(const uint32_t *input_array, uint32_t *intBuf) {
  uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t a = input_array[i];

  if (a <= 0xff) {
    intBuf[i] = 1;
    uint8_t b = static_cast<uint8_t>(a);
    memcpy((uint8_t *)(&intBuf[i]) + 1, &b, sizeof(uint8_t));
  } else if (a <= 0xffff) {
    intBuf[i] = sizeof(uint16_t);
    uint16_t s = static_cast<uint16_t>(a);
    memcpy((uint8_t *)(&intBuf[i]) + 1, &s, sizeof(uint16_t));
  } else {
    intBuf[i] = sizeof(uint32_t);
    memcpy((uint8_t *)(&intBuf[i]) + 1, &a, sizeof(uint32_t));
  }
}

__global__ void seq_compact(uint8_t *intBuf, const uint16_t dataSize, uint32_t *sizeBuf) {
  uint16_t writeIndex = 0;
  for (uint16_t i = 0; i < dataSize; ++i) {
    const uint16_t readIndex = i * 4;
    uint8_t size = intBuf[readIndex];
    memcpy(&intBuf[writeIndex], &intBuf[readIndex], size + 1);
    writeIndex += size + 1;
  }
  sizeBuf[0] = writeIndex;

  // zero out the rest of the buffer
  const uint32_t int_buf_size = (dataSize * sizeof(uint32_t)) + (dataSize * sizeof(uint8_t));
  memset(&intBuf[writeIndex], 0, int_buf_size - int_buf_size);
}

__global__ void bit_reduce_count(const uint32_t *input_array, uint32_t *intBuf, uint32_t *countBuf,
                                 const uint16_t dataCount) {
  extern __shared__ uint32_t sharedMem[];

  const uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
  const uint32_t a = input_array[i];
  uint8_t size = 0;
  if (a <= 0xff) {
    size = 1;
  } else if (a <= 0xffff) {
    size = sizeof(uint16_t);
  } else {
    size = sizeof(uint32_t);
  }

  sharedMem[threadIdx.x] = size;

  __syncthreads();

  // really dumb addition
  if (threadIdx.x == 1) {
    uint32_t total = 0;
    for (uint16_t i = 0; i < dataCount; i++) {
      total += sharedMem[i];
      sharedMem[i] = total;
    }
    countBuf[blockIdx.x] = total;
  }
  __syncthreads();

  // block comapct
  uint8_t* writeindex = (threadIdx.x + sharedMem[threadIdx.x] - size) + ((uint8_t*)&intBuf[(blockDim.x * blockIdx.x)]);
  //uint8_t* writeindex = (threadIdx.x + sharedMem[threadIdx.x] - size) + ((uint8_t*)&intBuf[0]);
  
  if (a <= 0xff) {
	*writeindex = 1;
    uint8_t b = static_cast<uint8_t>(a);
	memcpy(writeindex+1, &b, sizeof(uint8_t));
  } else if (a <= 0xffff) {
	*writeindex = sizeof(uint16_t);
    uint16_t s = static_cast<uint16_t>(a);
	memcpy(writeindex+1, &s, sizeof(uint16_t));
  } else {
	*writeindex = sizeof(uint32_t);
	memcpy(writeindex+1, &a, sizeof(uint32_t));
  }
  
}
/*
__global__ void count_compact(uint8_t *intBuf, const uint16_t blockSize, uint32_t *countBuf) {
	//read the countbuf into shared
	extern __shared__ uint8_t* sharedMem[2];

	const uint32_t i = threadIdx.x;
	const uint32_t block = blockDim.x * blockIdx.x;

	if (threadIdx.x == 1) {
		uint32_t count = countBuf[0];
		for (uint32_t j = 0; j < i; j++)
		{
			count += sharedMem[i];
		}
		count += ((i + 1) * blockSize);

		sharedMem[0] = &intBuf[count + 1]; //writeAddress
		sharedMem[1] = &intBuf[(i + 1) *(sizeof(uint32_t) * blockSize)]; //readAddress
	}
	__syncthreads();

	uint8_t
	//const uint8_t* readAddress = sharedMem[i] + ((i + 1) * 512)
//	const uint8_t* writeAddress = sharedMem[i] + ((i+1)*512)

}
*/

__global__ void move(uint8_t *buf, uint32_t dest, uint32_t source, uint16_t bytesEach, const bool wipe) {
	extern __shared__ uint8_t sharedMemT[];
	const uint32_t i = threadIdx.x;

	uint8_t *src = &buf[source];
	for (uint16_t j = 0; j < bytesEach; j++)
	{
		sharedMemT[(i*bytesEach) + j] = src[(i*bytesEach) + j];
		if (wipe){
			src[(i*bytesEach) + j] = 0;
		}
	}

	__syncthreads();

	uint8_t *d = &buf[dest];
	for (uint16_t j = 0; j < bytesEach; j++)
	{
		d[(i*bytesEach) + j] = sharedMemT[(i*bytesEach) + j];
	}
}

void Run_BitReduce(const dim3 a, const dim3 b, const uint32_t *input_array, uint32_t *intBuf,
	cudaStream_t &stream) {
	bit_reduce << <a, b, 0, stream >> >(input_array, intBuf);
}
  void Run_BitReduce_count(const dim3 a, const dim3 b, const uint32_t *input_array,
	  uint32_t *intBuf, uint32_t *countBuf, cudaStream_t &stream) {
	  bit_reduce_count << <a, b, b.x*sizeof(uint32_t), stream >> >(input_array, intBuf, countBuf, b.x);
  }

  void Run_seq_compact(const uint32_t *input_array, const uint16_t dataSize, uint32_t *sizeBuf,
                       cudaStream_t &stream) {
    const dim3 blocks(1, 1);
    seq_compact<<<blocks, blocks, 0, stream>>>((uint8_t *)input_array, dataSize, sizeBuf);
  }

  void Run_count_compact(const uint32_t count, const uint32_t blockSize, uint32_t *intBuf, uint32_t *countBuf, cudaStream_t &stream) {
	  const dim3 blocks(1, 1);
	  const dim3 threads(count-1, 1);
	//  count_compact << <blocks, threads, 0, stream >> >((uint8_t *)intBuf, blockSize, countBuf);
  }


  void Run_move(uint8_t *buf, uint32_t dest, uint32_t source, uint16_t threads, uint16_t bytesEach, const bool wipe, cudaStream_t &stream) {
	  const dim3 blocks(1, 1);
	  const dim3 threadsDim(threads, 1);
	  move << <blocks, threadsDim, bytesEach*threads, stream >> >(buf, dest, source, bytesEach, wipe);
  }
