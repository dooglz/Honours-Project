RWStructuredBuffer<uint> Buf;


cbuffer cbCS : register(b0)
{
  uint  j;
  uint  k;
  uint  yDimSize;
};

[numthreads(1, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
  unsigned int i, ixj; // Sorting partners: i and ixj
  i = tid.y + (tid.x * yDimSize);
  ixj = i^j;

  // The threads with the lowest ids sort the array.
  if ((ixj)>i) {
    if ((i&k) == 0) {
      // Sort ascending
      if (Buf[i]>Buf[ixj]) {
        // exchange(i,ixj);
        unsigned int temp = Buf[i];
        Buf[i] = Buf[ixj];
        Buf[ixj] = temp;
      }
    }
    if ((i&k) != 0) {
      // Sort descending
      if (Buf[i]<Buf[ixj]) {
        // exchange(i,ixj);
        unsigned int temp = Buf[i];
        Buf[i] = Buf[ixj];
        Buf[ixj] = temp;
      }
    }
  }
}
