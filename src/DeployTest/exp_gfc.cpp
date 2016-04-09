#include "exp_gfc.h"
#include "timer.h"
#include "utils.h"
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <stdio.h>

#define ull unsigned long long
//#define MAX (32 * 1024 * 1024)
#define WARPSIZE 32
//#define SIZE 64 * 1024 * 1024 // 64mb total
#define DEFAULTPOWER 18
//#define SIZE 512
Exp_Cuda_GFC::Exp_Cuda_GFC() : CudaExperiment(1, 1, "GFC float", "good compression") {}
Exp_Cuda_GFC::~Exp_Cuda_GFC() {}
static vector<cuda::CudaDevice> CtxDevices;
unsigned int Exp_Cuda_GFC::GetMinCu() { return 1; }
unsigned int Exp_Cuda_GFC::GetMax() { return 1; }

void RunGfCCompress(int blocks, int warpsperblock, cudaStream_t stream, int dimensionalityd,
                    unsigned long long *cbufd, unsigned char *dbufd, int *cutd, int *offd);

void RunGfCDECompress(int blocks, int warpsperblock, cudaStream_t stream, int dimensionalityd,
                      unsigned char *compressed_data_buffer_in, int *chunk_boundaries_buffer_in,
                      unsigned long long *uncompressed_data_buffer_out);

void Exp_Cuda_GFC::Shutdown() {}

void Exp_Cuda_GFC::Init(std::vector<cuda::CudaDevice> &devices) { CtxDevices = devices; }

void Exp_Cuda_GFC::Start(unsigned int num_runs, const std::vector<int> options) {
  if (CtxDevices.size() < GetMinCu() || CtxDevices.size() > GetMax()) {
    std::cout << "\n invalid number of devices\n";
    return;
  }

  uint16_t power;
  if (options.size() > 0) {
    power = options[0];
  } else {
    std::cout << "Power of numbers to compress?: (0 for default)" << std::endl;
    power = promptValidated<int, int>("Power: ", [](int i) { return (i >= 0 && i <= 256); });
  }
  if (power == 0) {
    power = DEFAULTPOWER;
  }
  const uint32_t SIZE = 1 << power;
  ResultFile r;
  r.name = "cudaGFCComp" + to_string(SIZE);
  r.headdings = { "compressTimeNS","RatioX1000" };

  checkCudaErrors(cudaSetDevice(CtxDevices[0].id));

  unsigned int runs = 0;
  running = true;
  should_run = true;
  float time_ms = 0;
  while (ShouldRun() && runs < num_runs) {
    unsigned long long time;
    unsigned int percentDone = (unsigned int)(floor(((float)runs / (float)num_runs) * 100.0f));
    std::cout << "\r" << Spinner(runs) << "\t" << runs << "\tPercent Done: " << percentDone << "%"<< std::flush;

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    cudaEvent_t event1;
    cudaEvent_t event2;
    checkCudaErrors(cudaEventCreate(&event1));
    checkCudaErrors(cudaEventCreate(&event2));
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

    // cout << sizeof(char)* ((doubles + 1) / 2 * 17) << endl;
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
    checkCudaErrors(cudaEventRecord(event1, stream));
    RunGfCCompress(blocks, WARPSIZE, stream, dimensionality, cbufl, (unsigned char *)dbufl, cutl,
                   offl);
    checkCudaErrors(cudaEventRecord(event2, stream));
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    
    checkCudaErrors(cudaEventElapsedTime(&time_ms, event1, event2));
    cudaDeviceSynchronize();
    getLastCudaError("GFC Kernel() execution failed.\n");
   // fprintf(stderr, "Compresison done\n");

    // transfer offsets back to CPU
    if (cudaSuccess !=
        cudaMemcpy(off, offl, sizeof(int) * blocks * warpsperblock, cudaMemcpyDeviceToHost))
      fprintf(stderr, "copying of off from device failed\n");
    /*
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
    */
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
      if (cudaSuccess != cudaMemcpy(dbuf + offset, dbufl + offset, sizeof(char) * off[i],
                                    cudaMemcpyDeviceToHost)) {

        fprintf(stderr, "copying of dbuf from device failed\n");
      }
      // std::copy(&dbuf[offset], (&dbuf[offset]) + off[i], std::ostream_iterator<char>(my_ss));
      // num = fwrite(&dbuf[offset], 1, off[i], pFile);
      // assert(off[i] == num);
    }
    // fclose(pFile);
    float ratio = (float)totalsize / (float)(SIZE * 8);
    /*cout << "Original size: " << readable_fs(SIZE * 8) << endl;
    cout << "Compressed size: " << readable_fs(totalsize) << endl;
    cout << "Ratio: " << ratio << endl;
    cout << "Time(ms) : " << time_ms << endl;
    cout << "Speed: " << ((double)SIZE / 1024.0 / 1024.0) / ((double)(time_ms * 0.001f)) << "MB/s"<< endl;
*/

    r.times.push_back({ (unsigned long long)(time_ms * 1000000.0f), (unsigned long long) (ratio *1000.0f) });

    /*
    // DECOMPRESS

    unsigned long long *uncompressed_data_buffer_out;
    if (cudaSuccess != cudaMalloc((void **)&uncompressed_data_buffer_out, sizeof(ull) * doubles))
      fprintf(stderr, "could not allocate cbufd\n");
    cudaDeviceSynchronize();

    checkCudaErrors(cudaEventRecord(event1, stream));
    RunGfCDECompress(blocks, WARPSIZE, stream, dimensionality,
                     (unsigned char *)dbufl,      // compressed_data_buffer_in
                     cutl,                        // chunk_boundaries_buffer_in
                     uncompressed_data_buffer_out // uncompressed_data_buffer_out
                     );
    checkCudaErrors(cudaEventRecord(event2, stream));
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
    getLastCudaError("GFC Decompression Kernel() execution failed.\n");
    fprintf(stderr, "Decompression done\n");

    checkCudaErrors(cudaEventElapsedTime(&time_ms, event1, event2));
    cout << "Time(ms): " << (time_ms) << endl;
    cout << "Speed: " << ((double)SIZE / 1024.0 / 1024.0) / ((double)(time_ms * 0.001f)) << "MB/s"
         << endl;
         */
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    checkCudaErrors(cudaEventDestroy(event1));
    checkCudaErrors(cudaEventDestroy(event2));
    checkCudaErrors(cudaStreamDestroy(stream));
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
    ++runs;
  }
  r.CalcAvg();
  r.PrintToCSV(r.name);
}
