__global__ void hello(unsigned int * buf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  buf[i] += 2;
}

void Run_NothingKernel(dim3 a, dim3 b, cudaStream_t stream, unsigned int*buf) {
  hello << <a, b, 0, stream >> > (buf);
}
