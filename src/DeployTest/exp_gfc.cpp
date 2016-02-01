#include "exp_gfc.h"
#include "timer.h"
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <stdio.h>

#define ull unsigned long long
//#define MAX (32 * 1024 * 1024)
#define WARPSIZE 32
#define SIZE 4 * 1024 * 1024 // 64mb total

Exp_Cuda_GFC::Exp_Cuda_GFC() : CudaExperiment(1, 2, "GFC float", "good compression") {}
Exp_Cuda_GFC::~Exp_Cuda_GFC() {}

unsigned int Exp_Cuda_GFC::GetMinCu() { return 2; }
unsigned int Exp_Cuda_GFC::GetMax() { return 2; }

void RunGfCCompress(int blocks, int warpsperblock, cudaStream_t stream, int dimensionalityd,
                    unsigned long long *cbufd, unsigned char *dbufd, int *cutd, int *offd);

void Exp_Cuda_GFC::Shutdown() {}

void Exp_Cuda_GFC::Start(unsigned int num_runs, const std::vector<int> options) {}

void Exp_Cuda_GFC::Init(std::vector<cuda::CudaDevice> &devices) {
  cudaDeviceReset();

  int blocks = 28;
  int warpsperblock = 18;
  int dimensionality = 1;
  // allocate CPU buffers
  double *cbuf = nullptr;
  char *dbuf = nullptr;
  int *cut = nullptr;
  int *off = nullptr;
  cbuf = new double[SIZE];               // uncompressed data
  dbuf = new char[(SIZE + 1) / 2 * 17];  // decompressed data
  cut = new int[blocks * warpsperblock]; // chunk boundaries
  off = new int[blocks * warpsperblock]; // offset table

  // int doubles = fread(cbuf, 8, MAX, stdin);
  int doubles = SIZE;
  for (size_t i = 0; i < SIZE; i++) {
    cbuf[i] = double(i) + (double(rand()) / (double(RAND_MAX)));
    // cbuf[i] = double(i);
  }
  // calculate required padding for last chunk
  int padding = ((doubles + WARPSIZE - 1) & -WARPSIZE) - doubles;
  doubles += padding;

  // determine chunk assignments per warp
  int per = (doubles + blocks * warpsperblock - 1) / (blocks * warpsperblock);
  if (per < WARPSIZE)
    per = WARPSIZE;
  per = (per + WARPSIZE - 1) & -WARPSIZE;
  int curr = 0, before = 0, d = 0;
  for (int i = 0; i < blocks * warpsperblock; i++) {
    curr += per;
    cut[i] = min(curr, doubles);
    if (cut[i] - before > 0) {
      d = cut[i] - before;
    }
    before = cut[i];
  }

  // set the pad values to ensure correct prediction
  if (d <= WARPSIZE) {
    for (int i = doubles - padding; i < doubles; i++) {
      cbuf[i] = 0;
    }
  } else {
    for (int i = doubles - padding; i < doubles; i++) {
      cbuf[i] = cbuf[(i & -WARPSIZE) - (dimensionality - i % dimensionality)];
    }
  }

  // allocate GPU buffers
  ull *cbufl;  // uncompressed data
  char *dbufl; // compressed data
  int *cutl;   // chunk boundaries
  int *offl;   // offset table
  if (cudaSuccess != cudaMalloc((void **)&cbufl, sizeof(ull) * doubles))
    fprintf(stderr, "could not allocate cbufd\n");
  if (cudaSuccess != cudaMalloc((void **)&dbufl, sizeof(char) * ((doubles + 1) / 2 * 17)))
    fprintf(stderr, "could not allocate dbufd\n");
  if (cudaSuccess != cudaMalloc((void **)&cutl, sizeof(int) * blocks * warpsperblock))
    fprintf(stderr, "could not allocate cutd\n");
  if (cudaSuccess != cudaMalloc((void **)&offl, sizeof(int) * blocks * warpsperblock))
    fprintf(stderr, "could not allocate offd\n");

  // copy buffer starting addresses (pointers) and values to constant memory
  /*
  if (cudaSuccess != cudaMemcpyToSymbol(dimensionalityd, &dimensionality, sizeof(int)))
    fprintf(stderr, "copying of dimensionality to device failed\n");
  if (cudaSuccess != cudaMemcpyToSymbol(cbufd, &cbufl, sizeof(void *)))
    fprintf(stderr, "copying of cbufl to device failed\n");
  if (cudaSuccess != cudaMemcpyToSymbol(dbufd, &dbufl, sizeof(void *)))
    fprintf(stderr, "copying of dbufl to device failed\n");
  if (cudaSuccess != cudaMemcpyToSymbol(cutd, &cutl, sizeof(void *)))
    fprintf(stderr, "copying of cutl to device failed\n");
  if (cudaSuccess != cudaMemcpyToSymbol(offd, &offl, sizeof(void *)))
    fprintf(stderr, "copying of offl to device failed\n");
    */

  // copy CPU buffer contents to GPU
  if (cudaSuccess != cudaMemcpy(cbufl, cbuf, sizeof(ull) * doubles, cudaMemcpyHostToDevice))
    fprintf(stderr, "copying of cbuf to device failed\n");
  if (cudaSuccess !=
      cudaMemcpy(cutl, cut, sizeof(int) * blocks * warpsperblock, cudaMemcpyHostToDevice))
    fprintf(stderr, "copying of cut to device failed\n");

  // CompressionKernel << <blocks, WARPSIZE*warpsperblock >> >();
  Timer t = Timer();
  RunGfCCompress(blocks, WARPSIZE, 0, dimensionality, cbufl, (unsigned char *)dbufl, cutl, offl);
  cudaDeviceSynchronize();
  t.Stop();
  getLastCudaError("GFC Kernel() execution failed.\n");
  fprintf(stderr, "done\n");

  // transfer offsets back to CPU
  if (cudaSuccess !=
      cudaMemcpy(off, offl, sizeof(int) * blocks * warpsperblock, cudaMemcpyDeviceToHost))
    fprintf(stderr, "copying of off from device failed\n");

  std::ostringstream my_ss;
  FILE *pFile;
  pFile = fopen("myfile.bin", "wb");

  // output header
  int num;
  int doublecnt = doubles - padding;
  num = fwrite(&blocks, 1, 1, pFile);
  assert(1 == num);
  num = fwrite(&warpsperblock, 1, 1, pFile);
  assert(1 == num);
  num = fwrite(&dimensionality, 1, 1, pFile);
  assert(1 == num);
  num = fwrite(&doublecnt, 4, 1, pFile);
  assert(1 == num);

  int blockcount = 0;
  uint32_t totalsize = 7;

  // output offset table
  for (int i = 0; i < blocks * warpsperblock; i++) {
    int start = 0;
    if (i > 0) {
      start = cut[i - 1];
    }
    off[i] -= ((start + 1) / 2 * 17);

    if (off[i] != 0) {
      ++blockcount;
      totalsize += 4;
      totalsize += off[i];
    }

    // std::copy(&off[i], (&off[i])+4, std::ostream_iterator<char>(my_ss));
    // num = fwrite(&off[i], 4, 1, pFile); // chunk's compressed size in bytes
    // assert(1 == num);
  }

  // output compressed data by chunk
  for (int i = 0; i < blocks * warpsperblock; i++) {
    int offset, start = 0;
    if (i > 0)
      start = cut[i - 1];
    offset = ((start + 1) / 2 * 17);
    // transfer compressed data back to CPU by chunk
    if (cudaSuccess !=
        cudaMemcpy(dbuf + offset, dbufl + offset, sizeof(char) * off[i], cudaMemcpyDeviceToHost)) {

      fprintf(stderr, "copying of dbuf from device failed\n");
    }
    // std::copy(&dbuf[offset], (&dbuf[offset]) + off[i], std::ostream_iterator<char>(my_ss));
    // num = fwrite(&dbuf[offset], 1, off[i], pFile);
    // assert(off[i] == num);
  }
  // fclose(pFile);

  cout << "Original size: " << readable_fs(SIZE * 8) << endl;
  cout << "Compressed size: " << " " << readable_fs(totalsize) << endl;
  cout << "Ratio: " << (float)totalsize / (float)(SIZE * 8) << endl;
  cout << "Time: " << t.Duration_NS() << endl;
  cout << "Speed: " << ((double)SIZE / 1024.0 / 1024.0) / ((double)t.Duration_NS() * 0.000000001)
       << "MB/s" << endl;

  delete (cbuf);
  delete (dbuf);
  delete (cut);
  delete (off);

  if (cudaSuccess != cudaFree(cbufl))
    fprintf(stderr, "could not deallocate cbufd\n");
  if (cudaSuccess != cudaFree(dbufl))
    fprintf(stderr, "could not deallocate dbufd\n");
  if (cudaSuccess != cudaFree(cutl))
    fprintf(stderr, "could not deallocate cutd\n");
  if (cudaSuccess != cudaFree(offl))
    fprintf(stderr, "could not deallocate offd\n");
}