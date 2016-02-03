#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <vector>
#include <assert.h>
#include <CL/opencl.h>

//
#include "ezOptionParser.hpp"
//
#include "utils.h"
#include "opencl_utils.h"
#include "cuda_utils.h"
#include "exp_cl_sort.h"
#include "exp_cl_sort2.h"
#include "exp_cuda_pingpong.h"
#include "exp_cuda_sort.h"
#include "exp_cpu_bit_reducer.h"
#include "exp_cuda_compression.h"
#include "exp_gfc.h"
using namespace std;

void Usage(ez::ezOptionParser &opt) {
  std::string usage;
  opt.getUsage(usage);
  std::cout << usage;
};

int main(int argc, const char *argv[]) {
  std::cout << "Hello World!\n";
  ez::ezOptionParser opt;

  opt.overview = "Demo of automatic usage message creation.";
  opt.syntax = "usage [OPTIONS]";
  opt.example = "usage -h\n\n";
  opt.footer = "Sam Serrels 2015\n";

  opt.add("", 0, 0, 0, "Display usage", "-h", "-help", "--help", "--usage");

  opt.add("", 0, 1, 0, "Batch Mode,-b [test]", "-b", "-batch", "--batch");
  opt.add("", 0, 1, ',', "Batch Platform", "-p");
  opt.add("", 0, -1, ',', "Batch Devices", "-d");
  opt.add("", 0, -1, ',', "Experiment options", "-e");

  opt.add("", 0, 1, ',', "iterations,", "-i", "--iterations");

  opt.add("", 0, 1, 0, "Output FileName, will overrite", "-f", "-file", "--outputfile");

  opt.parse(argc, argv);

  if (opt.isSet("-h")) {
    Usage(opt);
    return 0;
  }
  std::vector<std::string> badOptions;

  if (!opt.gotRequired(badOptions)) {
    for (size_t i = 0; i < badOptions.size(); ++i) {
      std::cerr << "ERROR: Missing required option " << badOptions[i] << ".\n\n";
    }
    Usage(opt);
    return 1;
  }

  if (!opt.gotExpected(badOptions)) {
    for (size_t i = 0; i < badOptions.size(); ++i) {
      std::cerr << "ERROR: Got unexpected number of arguments for option " << badOptions[i]
                << ".\n\n";
    }
    Usage(opt);
    return 1;
  }

  int selectedExp = 0;
  int selectedPlat = 0;
  std::vector<int> selectedDevices;
  std::vector<int> expOptions;

  bool batch = false;
  if (opt.isSet("-b")) {
    batch = true;
    cout << "Batch mode selected" <<endl;

    opt.get("-b")->getInt(selectedExp);
    opt.get("-p")->getInt(selectedPlat);
    opt.get("-d")->getInts(selectedDevices);
    cout << "Experiment: " << selectedExp << " selectedPlat: " << selectedPlat << " selectedDevices: #" << selectedDevices.size() << endl;

    if (opt.isSet("-e")) {
      cout << "Experiemnt options ";
      opt.get("-e")->getInts(expOptions);
      for (size_t j = 0; j < expOptions.size(); ++j) {
        std::cout << " " << expOptions[j];
      }
      std::cout << endl;
    }

    std::cout << endl;
  }

  uint16_t iterations = 100;
  if (opt.isSet("-i")) {
    int a;
    opt.get("-b")->getInt(a);
    iterations = a;
  }

  // init cl
  cl::Init();
  if (!batch){
    cl::PrintInfo();
  }
  std::vector<cl::CLDevice *> devices;
  cl::GetRecommendedDevices(7, devices);

  // init cuda
  cuda::Init();
  if (!batch){
    cuda::PrintInfo();
  }
  // start menu system
  std::cout << "\nRecommended devices:\n";
  for (auto dev : devices) {
    std::cout << dev->short_name << "\n";
  }
  // create list of tests
  Experiment *exps[7];
  exps[0] = new Sort();
  exps[1] = new Sort2();
  exps[2] = new CudaSort();
  exps[3] = new Exp_Cuda_PingPong();
  exps[4] = new Exp_Cpu_BitReducer();
  exps[5] = new Exp_Cuda_Compression();
  exps[6] = new Exp_Cuda_GFC();
  bool run = true;

  // main loop

  if (batch) {
    if (selectedExp < 0 || selectedExp > 7) { // todo better validation
      cout << "Invalid Test" << std::endl;
      return 1;
    }
    exps[selectedExp - 1]->Init2(true, selectedPlat, selectedDevices);
    exps[selectedExp - 1]->Start(300, expOptions);
  } else {
    while (run) {
      selectedExp = 0;
      // print tests
      cout << "\nAvaialble Experiments:" << std::endl;
      cout << "\t0\tQuit" << std::endl;
      for (size_t i = 0; i < 7; i++) {
        cout << "\t" << i + 1 << "\t" << exps[i]->name << "\t\t" << exps[i]->description
             << std::endl;
      }
      selectedExp = promptValidated<int, int>("Choose an Experiment: ",
                                              [](int i) { return (i >= 0 && i <= 7); });
      if (selectedExp == 0) {
        run = false;
        break;
      }
      selectedExp = selectedExp - 1;
      exps[selectedExp]->Init2(false, selectedPlat, selectedDevices);
      exps[selectedExp]->Start(300, expOptions);
    }
  }

  delete[] cl::CLdevices;
  delete[] cuda::CudaDevices;

  for (auto e : exps) {
    delete e;
  }

  std::cout << "\nbye!\n";
}