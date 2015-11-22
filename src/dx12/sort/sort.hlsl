RWStructuredBuffer<uint> Buf;


cbuffer cbCS : register(b0)
{
  uint  j;
  uint  k;
};

/*
[numthreads(14, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
	// Hello, World!
	uint gOutputString[14] = { 72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33, 0 };

	// Write message
	Buf[tid.x] = gOutputString[tid.x];
}
*/

[numthreads(1, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{

  //Buf[tid.x] = tid.x + j + k;
  Buf[tid.x] = Buf[tid.x];
  /*
  unsigned int i, ixj; // Sorting partners: i and ixj
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  // The threads with the lowest ids sort the array.
  if ((ixj)>i) {
    if ((i&k) == 0) {
      // Sort ascending
      if (dev_values[i]>dev_values[ixj]) {
        // exchange(i,ixj);
        unsigned int temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k) != 0) {
      // Sort descending
      if (dev_values[i]<dev_values[ixj]) {
        // exchange(i,ixj);
        unsigned int temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
  */
}

