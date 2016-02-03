#include "exp_cpu_bit_reducer.h"

#include "utils.h"
#include "timer.h"
#include "limits.h"
#include <chrono>
#include <thread>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <string>
#include <cstring>

using namespace std;

Exp_Cpu_BitReducer::Exp_Cpu_BitReducer()
    : Experiment(1, 4, "Cpu Bit Reducer", "wost compression ever") {}
Exp_Cpu_BitReducer::~Exp_Cpu_BitReducer() {}

void Exp_Cpu_BitReducer::Init2(bool batch, int selectedPlat, std::vector<int> selectedDevices) {}
void Exp_Cpu_BitReducer::Shutdown() {}

#define COUNT 1024
void Exp_Cpu_BitReducer::Start(unsigned int num_runs, const std::vector<int> options) {
  std::cout << "\n Cpu Bit Reducer\n";
  srand((uint32_t)time(NULL));

  // create input data
  uint32_t *inData = new uint32_t[COUNT];
  const size_t dataSize = (COUNT) * sizeof(uint32_t);
  for (cl_uint i = 0; i < COUNT; i++) {
    uint32_t x = 0;
    inData[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
    inData[i] = (x << 14) | ((uint32_t)rand() & 0x3FFF);
  }

  Timer tm;
  tm.Start();
  uint8_t *outData = new uint8_t[dataSize + COUNT];

  size_t chars = 0;
  size_t shorts = 0;
  size_t ints = 0;

  size_t writeIndex = 0;
  for (size_t i = 0; i < COUNT; i++) {
    uint32_t a = inData[i];
    if (a <= UCHAR_MAX) {
      outData[writeIndex] = 1;
      ++writeIndex;

      uint8_t b = static_cast<uint8_t>(a);
      outData[writeIndex] = b;
      ++writeIndex;
      ++chars;
    } else if (a <= USHRT_MAX) {
      outData[writeIndex] = sizeof(uint16_t);
      ++writeIndex;

      uint16_t s = static_cast<uint16_t>(a);
      memcpy(&outData[writeIndex], &s, sizeof(uint16_t));
      writeIndex += sizeof(uint16_t);
      ++shorts;
    } else {
      outData[writeIndex] = sizeof(uint32_t);
      ++writeIndex;

      memcpy(&outData[writeIndex], &a, sizeof(uint32_t));
      writeIndex += sizeof(uint32_t);
      ++ints;
    }
  }
  // alright, now we can comrpess hte final outbuffer
  uint8_t *compressed = new uint8_t[writeIndex];
  memcpy(&compressed[0], &outData[0], writeIndex);

  // decompress
  uint32_t *decompressed = new uint32_t[COUNT];
  size_t outIndex = 0;

  for (size_t i = 0; i < writeIndex;) {
    uint8_t size = compressed[i];
	++i;

    if (size == 1) {
		decompressed[outIndex] = static_cast<uint32_t>(compressed[i]);
    } else if (size == sizeof(uint16_t)) {
		uint16_t s;
		memcpy(&s, &compressed[i], sizeof(uint16_t));
		decompressed[outIndex] = static_cast<uint32_t>(s);
    } else {
		uint32_t b;
		memcpy(&b, &compressed[i], sizeof(uint32_t));
		decompressed[outIndex] = static_cast<uint32_t>(b);
    }

	i += size;
	++outIndex;
  }
  tm.Stop();

  const float ratio = (float)writeIndex / (float)dataSize;
  const bool b = CheckArraysEqual(decompressed, inData, COUNT);

  cout << "Validation " << (b ? "passed" : "failed") << " Ratio: " << ratio << " t: " << tm.Duration_NS() << endl;

  delete[] inData;
  delete[] outData;
  delete[] compressed;
  delete[] decompressed;

  cout << "\n Exp finished\n";
  running = false;
};
