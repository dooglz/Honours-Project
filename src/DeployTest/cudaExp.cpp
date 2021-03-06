#include "utils.h"
#include "cudaExp.h"
#include "cuda_utils.h"
#include <thread>
#include <fstream>
#include <iostream>
using namespace std;

CudaExperiment::CudaExperiment(const unsigned int minCu, const unsigned int maxCU,
                               const std::string &name, const std::string &description)
    : Experiment(minCu, maxCU, name, description) {
  running = false;
  should_run = false;
}

CudaExperiment::~CudaExperiment() { Shutdown(); }

void CudaExperiment::Init(std::vector<cuda::CudaDevice> &devices){};

void CudaExperiment::Init2(bool batch, int selectedPlat, std::vector<int> selectedDevices) {
  if (!batch) {
    unsigned int selectedDev = 0;
    unsigned int num_selected = 0;
    std::vector<cuda::CudaDevice> sel_devices;

    bool *selected = new bool[5];
    std::fill(selected, selected + sizeof(selected) / sizeof(bool), false);
    selectedDev = 2; // just to get us in the loop.
    while (selectedDev != 0 && selectedDev != 1) {
      if (num_selected == 0) {
        cout << "Avaialble Devices:" << std::endl;
        cout << "\t0\tCancel\n\t1\tUse Reccomended" << std::endl;
      } else {
        cout << "Avaialble Devices:" << std::endl;
        cout << "\t0\tCancel\n\t1\tDone" << std::endl;
      }
      for (size_t i = 0; i < cuda::total_num_devices; i++) {
        cout << "\t" << i + 2 << "\t" << cuda::CudaDevices[i].short_name
             << "\t CU:" << cuda::CudaDevices[i].computeUnits;
        if (selected[i] == true) {
          cout << "\t Selected" << std::endl;
        } else {
          cout << std::endl;
        }
      }
      unsigned int offset = cuda::total_num_devices;
      selectedDev = promptValidated<int, int>(
          "Choose a Device: ", [offset](unsigned int j) { return (j >= 0 && j <= (2 + offset)); });
      if (selectedDev > 1) {
        selected[selectedDev - 2] = !selected[selectedDev - 2];
        if (selected[selectedDev - 2] == true) {
          num_selected++;
        } else {
          num_selected--;
        }
      }
    }

    for (size_t i = 0; i < 5; i++) {
      if (selected[i] == true) {
        sel_devices.push_back(cuda::CudaDevices[i]);
      }
    }
    delete[] selected;

    if (selectedDev == 1) {
      if (num_selected == 0) {
        // use defaults
        Init(sel_devices);

        exit(1);
      }
      else {
        // load selected
        cout << "Cuda experiment initialised with " << sel_devices.size() << " devices: ";
        for (auto d : sel_devices) {
          cout << d.short_name << "(" << d.id << "), ";
        }
        cout << endl;

        Init(sel_devices);
      }
    }
    else {
      // Cancelled out
      return;
    }
  } else {
    // batch init
    std::vector<cuda::CudaDevice> sel_devices;
    for (auto d : selectedDevices) {
      sel_devices.push_back(cuda::CudaDevices[d]);
    }
    if (sel_devices.size() == 0) {
      sel_devices.push_back(cuda::CudaDevices[0]);
    }
    cout << "Cuda experiment initialised with " << sel_devices.size() << " devices: ";
    for (auto d : sel_devices) {
      cout << d.short_name << "(" << d.id << "), ";
    }
    cout << endl;
    Init(sel_devices);
  }
};
