#define WARPSIZE 32
#include<stdio.h>
#include<stdlib.h>

__global__ void CompressionKernel(int dimensionalityd, unsigned long long *cbufd,
                                  unsigned char *dbufd, int *cutd, int *offd) {
  register int offset, code, bcount, tmp, off, beg, end, lane, warp, iindex, lastidx, start, term;
  register unsigned long long diff, prev;
  __shared__ int ibufs[32 * (3 * WARPSIZE / 2)]; // shared space for prefix sum

  // index within this warp
  lane = threadIdx.x & 31;
  // index within shared prefix sum array
  iindex = threadIdx.x / WARPSIZE * (3 * WARPSIZE / 2) + lane;
  ibufs[iindex] = 0;
  iindex += WARPSIZE / 2;
  lastidx = (threadIdx.x / WARPSIZE + 1) * (3 * WARPSIZE / 2) - 1;
  // warp id
  warp = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;
  // prediction index within previous subchunk
  offset = WARPSIZE - (dimensionalityd - lane % dimensionalityd) - lane;

  // determine start and end of chunk to compress
  start = 0;
  if (warp > 0)
    start = cutd[warp - 1];
  term = cutd[warp];
  off = ((start + 1) / 2 * 17);

  prev = 0;
  for (int i = start + lane; i < term; i += WARPSIZE) {
    // calculate delta between value to compress and prediction
    // and negate if negative
    diff = cbufd[i] - prev;
    code = (diff >> 60) & 8;
    if (code != 0) {
      diff = -diff;
    }

    // count leading zeros in positive delta
    bcount = 8 - (__clzll(diff) >> 3);
    if (bcount == 2)
      bcount = 3; // encode 6 lead-zero bytes as 5

    // prefix sum to determine start positions of non-zero delta bytes
    ibufs[iindex] = bcount;
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex - 1];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex - 2];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex - 4];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex - 8];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex - 16];
    __threadfence_block();

    // write out non-zero bytes of delta to compressed buffer
    beg = off + (WARPSIZE / 2) + ibufs[iindex - 1];
    end = beg + bcount;
    for (; beg < end; beg++) {
      dbufd[beg] = diff;
      diff >>= 8;
    }

    if (bcount >= 3)
      bcount--; // adjust byte count for the dropped encoding
    tmp = ibufs[lastidx];
    code |= bcount;
    ibufs[iindex] = code;
    __threadfence_block();

    // write out half-bytes of sign and leading-zero-byte count (every other thread
    // writes its half-byte and neighbor's half-byte)
    if ((lane & 1) != 0) {
      dbufd[off + (lane >> 1)] = ibufs[iindex - 1] | (code << 4);
    }
    off += tmp + (WARPSIZE / 2);

    // save prediction value from this subchunk (based on provided dimensionality)
    // for use in next subchunk
    prev = cbufd[i + offset];
  }

  // save final value of off, which is total bytes of compressed output for this chunk
  if (lane == 31)
    offd[warp] = off;
}

void RunGfCCompress(int blocks, int warpsperblock, cudaStream_t stream, int dimensionalityd,
                    unsigned long long *cbufd, unsigned char *dbufd, int *cutd, int *offd) {
  CompressionKernel << <blocks, WARPSIZE * warpsperblock, 0,stream >> >(dimensionalityd, cbufd, dbufd, cutd, offd);
  // cudaDeviceSynchronize();
}

__global__ void DecompressionKernel(int dimensionalityd, unsigned char *compressed_data_buffer_in,
                                    int *chunk_boundaries_buffer_in,
                                    unsigned long long *uncompressed_data_buffer_out) {
  register int offset, code, bcount, off, beg, end, lane, warp, iindex, lastidx, start, term;
  register unsigned long long diff, prev;
  __shared__ int ibufs[32 * (3 * WARPSIZE / 2)];


  // index within this warp
  lane = threadIdx.x & 31;
  // index within shared prefix sum array
  iindex = threadIdx.x / WARPSIZE * (3 * WARPSIZE / 2) + lane;
  ibufs[iindex] = 0;
  iindex += WARPSIZE / 2;
  lastidx = (threadIdx.x / WARPSIZE + 1) * (3 * WARPSIZE / 2) - 1;
  // warp id
  warp = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;
  // prediction index within previous subchunk
  offset = WARPSIZE - (dimensionalityd - lane % dimensionalityd) - lane;

  // determine start and end of chunk to decompress
  start = 0;
  if (warp > 0)
    start = chunk_boundaries_buffer_in[warp - 1];
  term = chunk_boundaries_buffer_in[warp];
  off = ((start + 1) / 2 * 17);

  prev = 0;
  for (int i = start + lane; i < term; i += WARPSIZE) {
    // read in half-bytes of size and leading-zero count information
 
    if ((lane & 1) == 0) {
      code = compressed_data_buffer_in[off + (lane >> 1)];

//4352
     // printf(" %i ", start);
      return;
      ibufs[iindex] = code; //THIS line is crashing
      return;
      ibufs[iindex + 1] = code >> 4;

    }
    return;
    off += (WARPSIZE / 2);
    __threadfence_block();
    code = ibufs[iindex];

    bcount = code & 7;
    if (bcount >= 2)
      bcount++;
 
    // calculate start positions of compressed data
    ibufs[iindex] = bcount;
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex - 1];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex - 2];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex - 4];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex - 8];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex - 16];
    __threadfence_block();

    // read in compressed data (the non-zero bytes)
    beg = off + ibufs[iindex - 1];
    off += ibufs[lastidx];
    end = beg + bcount - 1;
    diff = 0;
    for (; beg <= end; end--) {
      diff <<= 8;
      diff |= compressed_data_buffer_in[end];
    }

    // negate delta if sign bit indicates it was negated during compression
    if ((code & 8) != 0) {
      diff = -diff;
    }

    // write out the uncompressed word
    uncompressed_data_buffer_out[i] = prev + diff;
    __threadfence_block();

    // save prediction for next subchunk
    prev = uncompressed_data_buffer_out[i + offset];
  }
}

void RunGfCDECompress(int blocks, int warpsperblock, cudaStream_t stream, int dimensionalityd,
                      unsigned char *compressed_data_buffer_in, int *chunk_boundaries_buffer_in,
                      unsigned long long *uncompressed_data_buffer_out) {
  DecompressionKernel<<<blocks, WARPSIZE * warpsperblock>>>(
      dimensionalityd, compressed_data_buffer_in, chunk_boundaries_buffer_in,
      uncompressed_data_buffer_out);
  // cudaDeviceSynchronize();
}
