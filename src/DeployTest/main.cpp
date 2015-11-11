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
#include "sort.h"
#include "cudaSort.h"
#include "test.h"

using namespace std;

enum STATE {
  CHOOSE,
  LOADIN,
  LOADOUT,
  WORK,
  CHOOSEP,
  CHOOSED,
};

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
  opt.add("", 0, 3, ',', "Batch Mode,-b [test],[platform],[device]", "-b", "-batch", "--batch");
  opt.add("", 0, 1, ',', "iterations,", "-i", "--iterations");
  opt.add("", 0, -1, ',', "Experiment options", "-e");
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

  unsigned int selectedExp = 0;
  unsigned int selectedPlat = 0;
  unsigned int selectedDev = 0;

  bool batch = false;
  if (opt.isSet("-b")) {
    batch = true;
    cout << "Batch mode selected";
    std::vector<int> list;
    opt.get("-b")->getInts(list);
    for (size_t j = 0; j < list.size(); ++j) {
      std::cout << " " << list[j];
    }
    selectedExp = list[0];
    selectedPlat = list[1];
    selectedDev = list[2];
    std::cout << endl;
  }

  uint16_t iterations = 100;
  if (opt.isSet("-i")) {
    int a;
    opt.get("-b")->getInt(a);
    iterations = a;
  }

  std::vector<int> expOptions;
  if (opt.isSet("-e")) {
    cout << "Experiemnt options ";
    opt.get("-e")->getInts(expOptions);
    for (size_t j = 0; j < expOptions.size(); ++j) {
      std::cout << " " << expOptions[j];
    }
    std::cout << endl;
  }

  // init cl
  cl::Init();
  cl::PrintInfo();
  std::vector<cl::CLDevice *> devices;
  cl::GetRecommendedDevices(7, devices);

  // init cuda
  cuda::Init();
  cuda::PrintInfo();

  // start menu system
  std::cout << "\nRecommended devices:\n";
  for (auto dev : devices) {
    std::cout << dev->short_name << "\n";
  }
  // create list of tests
  Experiment *exps[2];
  exps[0] = new Sort();
  exps[1] = new CudaSort();
  bool run = true;

  // main loop

  if (batch) {
    if (selectedExp < 0 || selectedExp > 3) { // todo better validation
      cout << "Invalid Test" << std::endl;
      return 1;
    }
    exps[selectedExp]->Init2(true);
  } else {
    while (run) {
      selectedExp = 0;
      selectedPlat = 0;
      selectedDev = 0;
      // print tests
      cout << "\nAvaialble Experiments:" << std::endl;
      cout << "\t0\tQuit" << std::endl;
      for (size_t i = 0; i < 2; i++) {
        cout << "\t" << i + 1 << "\t" << exps[i]->name << "\t\t" << exps[i]->description
             << std::endl;
      }
      selectedExp = promptValidated<int, int>("Choose an Experiment: ",
                                              [](int i) { return (i >= 0 && i <= 3); });
      if (selectedExp == 0) {
        run = false;
        break;
      }
      selectedExp = selectedExp - 1;
      exps[selectedExp]->Init2(false);
      exps[selectedExp]->Start(300, expOptions);
    }
  }

  delete cl::devices;
  delete cuda::devices;
  for (auto e : exps) {
    delete e;
  }

  std::cout << "\nbye!\n";
}