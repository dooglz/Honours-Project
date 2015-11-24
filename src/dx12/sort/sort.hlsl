RWStructuredBuffer<uint> Buf;


cbuffer cbCS : register(b0)
{
  uint  j;
  uint  k;
  uint  yDimSize;
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
/*
[numthreads(1, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
//void CSMain(uint3 tid : SV_GroupID)
{

  //Buf[tid.x] = tid.x + j + k;
  //Buf[tid.x] = Buf[tid.x];
  
  unsigned int i, ixj; // Sorting partners: i and ixj
  i = tid.y + (tid.x * yDimSize);
  ixj = i^j;

 // Buf[i] = yDimSize;
 // return;

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
*/

[numthreads(1, 1, 1)] 
void CSMain(uint3 tid : SV_GroupID) {
  uint i = tid.y + (tid.x * yDimSize);
  uint ix = i*2;
  if (j) {
    if (i == 0) {
      return;
    }
    if (Buf[ix - 1] > Buf[ix]) {
      uint temp = Buf[ix];
      Buf[ix] = Buf[ix - 1];
      Buf[ix - 1] = temp;
    }
  } else {
    if (Buf[ix + 1] < Buf[ix]) {
      uint temp = Buf[ix];
      Buf[ix] = Buf[ix + 1];
      Buf[ix + 1] = temp;
    }
  }
}
