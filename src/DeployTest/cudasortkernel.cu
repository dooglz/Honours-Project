__global__ 
void hello(char *a, int *b) 
{
  a[threadIdx.x] += b[threadIdx.x];
}

void my_cuda_func(dim3 a, dim3 b, char *ab, int * bd){
  hello << <a, b >> >(ab, bd);
  //cudaDeviceSynchronize();
}