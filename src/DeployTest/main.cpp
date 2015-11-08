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
  STATE st = CHOOSE;
  // create list of tests
  Experiment *exps[3];
  exps[0] = new Test();
  exps[1] = new Sort();
  exps[2] = new CudaSort();
  bool run = true;

  // main loop

  while (run) {
    switch (st) {
    case CHOOSE: {
      if (!batch) {
        selectedExp = 0;
        selectedPlat = 0;
        selectedDev = 0;
        // print tests
        cout << "\nAvaialble Experiments:" << std::endl;
        cout << "\t0\tQuit" << std::endl;
        for (size_t i = 0; i < 3; i++) {
          cout << "\t" << i + 1 << "\t" << exps[i]->name << "\t\t" << exps[i]->description
               << std::endl;
        }
        // double num = promptValidated<double, double>("Enter any number: ");
        // cout << "The number is " << num << endl << endl;
        selectedExp = promptValidated<int, int>("Choose an Experiment: ",
                                                [](int i) { return (i >= 0 && i <= 3); });
      } else {
        if (selectedExp < 0 || selectedExp > 3) { // todo betterv alidation
          cout << "Invalid Test" << std::endl;
          return 1;
        }
      }

      if (selectedExp == 0) {
        run = false;
        break;
      }
      selectedExp = selectedExp - 1;
      st = CHOOSEP;
    } break;
    case CHOOSEP: {
      if (!batch) {
        cout << "Experiment requires a Minimum number of " << exps[selectedExp]->minCu
             << " devices, and a maximum of " << exps[selectedExp]->maxCU << std::endl;
        cout << "Avaialble Platforms:" << std::endl;
        cout << "\t0\tCancel\n\t1\tUse Reccomended" << std::endl;

        for (size_t i = 0; i < cl::total_num_platforms; i++) {
          cout << "\t" << i + 2 << "\t" << cl::platforms[i].short_name
               << "\t Devices:" << cl::platforms[i].num_devices << std::endl;
        }

        selectedPlat =
            promptValidated<unsigned int, unsigned int>("Choose an Platform: ", [](unsigned int j) {
              return (j >= 0 && j <= (2 + cl::total_num_platforms));
            });
      } else {
        if (selectedExp < 0 ||
            selectedExp > (2 + cl::total_num_platforms)) { // todo better validation
          cout << "Invalid Platform" << std::endl;
          return 1;
        }
      }
      if (selectedPlat == 0) {
        st = CHOOSE;
        break;
      }
      selectedPlat = selectedPlat - 2;
      st = CHOOSED;
    } break;
    case CHOOSED: {
      unsigned int num_selected = 0;
      std::vector<cl::CLDevice> sel_devices;
      if (!batch) {
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
          for (size_t i = 0; i < cl::platforms[selectedPlat].num_devices; i++) {
            cout << "\t" << i + 2 << "\t" << cl::platforms[selectedPlat].devices[i]->short_name
                 << "\t CU:" << cl::platforms[selectedPlat].devices[i]->computeUnits;
            if (selected[i] == true) {
              cout << "\t Selected" << std::endl;
            } else {
              cout << std::endl;
            }
          }
          unsigned int offset = cl::platforms[selectedPlat].num_devices;
          selectedDev = promptValidated<int, int>("Choose a Device: ", [offset](unsigned int j) {
            return (j >= 0 && j <= (2 + offset));
          });
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
            sel_devices.push_back(*cl::platforms[selectedPlat].devices[i]);
          }
        }
        delete[] selected;
      } else {
        // batch
        num_selected = 1;
        sel_devices.push_back(*cl::platforms[selectedPlat].devices[selectedDev - 2]);
        selectedDev = 1;
      }
      if (selectedDev == 1) {
        if (num_selected == 0) {
          // use defaults
        } else {
          // load selected
          cl_context context;
          std::vector<cl_command_queue> cmd_queue;
          GetContext(sel_devices, context, cmd_queue);
          exps[selectedExp]->Init(context, cmd_queue, sel_devices, cl::platforms[selectedPlat]);
        }
        st = LOADIN;
        break;
      } else {
        // Cancelled out
        st = CHOOSE;
        break;
      }
    } break;
    case LOADOUT:
      cout << DASH50 << "\n Finished" << CLEARN << std::endl;
      st = CHOOSE;
      break;
    case LOADIN:
      cout << DASH50 << CLEARN "Starting Experiment \n" << DASH50 << std::endl;
      exps[selectedExp]->Start(100, expOptions);
      st = LOADOUT;
      break;
    }
  }
  for (auto e : exps) {
    delete e;
  }

  std::cout << "\nbye!\n";
}