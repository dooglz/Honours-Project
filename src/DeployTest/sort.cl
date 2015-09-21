
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