

























__kernel void bitonicSort(__global uint *theArray, const uint stage, const uint passOfStage,
                          const uint width // amount of items in the array
                          ) {
  uint sortIncreasing = 1;
  uint threadId = get_global_id(0);

  uint pairDistance = 1 << (stage - passOfStage);
  uint blockWidth = 2 * pairDistance;

  uint leftId = (threadId % pairDistance) + (threadId / pairDistance) * blockWidth;

  uint rightId = leftId + pairDistance;

  uint leftElement = theArray[leftId];
  uint rightElement = theArray[rightId];

  uint sameDirectionBlockWidth = 1 << stage;

  if ((threadId / sameDirectionBlockWidth) % 2 == 1) {
    sortIncreasing = 1 - sortIncreasing;
  }

  uint greater;
  uint lesser;
  if (leftElement > rightElement) {
    greater = leftElement;
    lesser = rightElement;
  } else {
    greater = rightElement;
    lesser = leftElement;
  }

  if (sortIncreasing) {
    theArray[leftId] = lesser;
    theArray[rightId] = greater;
  } else {
    theArray[leftId] = greater;
    theArray[rightId] = lesser;
  }
}

__kernel void ParallelSelection_Local(__global const uint *in, __global uint *out,
                                      __local uint *aux) {
  int i = get_local_id(0);    // index in workgroup
  int wg = get_local_size(0); // workgroup size = block size

  // Move IN, OUT to block start
  int offset = get_group_id(0) * wg;
  in += offset;
  out += offset;

  // Load block in AUX[WG]
  uint iData = in[i];
  aux[i] = iData;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Find output position of iData
  uint iKey = iData;
  int pos = 0;
  for (int j = 0; j < wg; j++) {
    uint jKey = aux[j];
    bool smaller = (jKey < iKey) || (jKey == iKey && j < i); // in[j] < in[i] ?
    pos += (smaller) ? 1 : 0;
  }

  // Store output
  out[pos] = iData;
}
