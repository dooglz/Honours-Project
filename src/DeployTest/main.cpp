#include <iostream>
#include <vector>
#include <assert.h>
#include <CL/opencl.h>
//
#include "utils.h"
#include "opencl_utils.h"
#include "sort.h"
#include "test.h"
// for sleep
#include <chrono>
#include <thread>
#include <string>
#include <sstream>
#include "ezOptionParser.hpp"

using namespace std;

enum STATE {
  CHOOSE,
  LOADIN,
  LOADOUT,
  WORK,
  CHOOSEP,
  CHOOSED,
};

// Initialise OpenCL
void initialise_opencl(vector<cl_platform_id> &platforms, vector<cl_device_id> &devices,
                       cl_context &context, cl_command_queue &cmd_queue) {
  // Status of OpenCL calls
  cl_int status;

  // Get the number of platforms
  cl_uint num_platforms;
  status = clGetPlatformIDs(0, nullptr, &num_platforms);
  assert(status != CL_SUCCESS);

  // Resize vector to store platforms
  platforms.resize(num_platforms);

  // Fill in platform vector
  status = clGetPlatformIDs(num_platforms, &platforms[0], nullptr);
  assert(status != CL_SUCCESS);

  // Assume platform 0 is the one we want to use
  // Get devices for platform 0
  cl_uint num_devices;
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  assert(status != CL_SUCCESS);

  // Resize vector to store devices
  devices.resize(num_devices);
  // Fill in devices vector
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, &devices[0], nullptr);
  assert(status != CL_SUCCESS);

  // Create a context
  context = clCreateContext(nullptr, num_devices, &devices[0], nullptr, nullptr, &status);

  // Create a command queue
  cmd_queue = clCreateCommandQueue(context, devices[0], 0, &status);
}

template <class T> bool lexical_cast(T &result, const std::string &str) {
  std::stringstream s(str);
  return (s >> result && s.rdbuf()->in_avail() == 0);
}

template <class T, class U>
T promptValidated(const std::string &message,
                  std::function<bool(U)> condition = [](...) { return true; }) {
  T input;
  std::string buf;
  while (!(std::cout << message,
           std::getline(std::cin, buf) && lexical_cast<T>(input, buf) && condition(input))) {
    if (std::cin.eof())
      throw std::runtime_error("End of file reached!");
  }
  return input;
}
template <class T, class U>
T nopromptValidated(std::function<bool(U)> condition = [](...) { return true; }) {
  T input;
  std::string buf;
  while (!(std::getline(std::cin, buf) && lexical_cast<T>(input, buf) && condition(input))) {
    if (std::cin.eof())
      throw std::runtime_error("End of file reached!");
  }
  return input;
}

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

  opt.add("", 0, 3, ',', "Batch Mode,-b [test],[platform],[device]", "-b", "-batch",
          "--batch");

  opt.add("", 0, 1, 0, "Output FileName, will overrite", "-f", "-file",
          "--outputfile");

  opt.parse(argc, argv);

  if (opt.isSet("-h")) {
    Usage(opt);
    return 0;
  }
  std::vector<std::string> badOptions;
  int i;
  if (!opt.gotRequired(badOptions)) {
    for (i = 0; i < badOptions.size(); ++i) {
      std::cerr << "ERROR: Missing required option " << badOptions[i] << ".\n\n";
    }
    Usage(opt);
    return 1;
  }

  if (!opt.gotExpected(badOptions)) {
    for (i = 0; i < badOptions.size(); ++i) {
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
    for (int j = 0; j < list.size(); ++j) {
      std::cout << " " << list[j];
    }
    selectedExp = list[0];
    selectedPlat = list[1];
    selectedDev = list[2];
    std::cout << endl;
  }

  // init cl
  cl::Init();
  cl::PrintInfo();
  std::vector<cl::Device *> devices;
  cl::GetRecommendedDevices(7, devices);

  // start menu system
  std::cout << "\nRecommended devices:\n";
  for (auto dev : devices) {
    std::cout << dev->short_name << "\n";
  }
  STATE st = CHOOSE;
  // create list of tests
  Experiment *exps[2];
  exps[0] = new Test();
  exps[1] = new Sort();
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
        for (size_t i = 0; i < 2; i++) {
          cout << "\t" << i + 1 << "\t" << exps[i]->name << "\t" << exps[i]->description << std::endl;
        }
        // double num = promptValidated<double, double>("Enter any number: ");
        // cout << "The number is " << num << endl << endl;
        selectedExp = promptValidated<int, int>("Choose an Experiment: ",
          [](int i) { return (i >= 0 && i <= 2); });
      }
      else{
        if (selectedExp <0 || selectedExp>2){ //todo betterv alidation
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
      }else{
        if (selectedExp < 0 || selectedExp > (2 + cl::total_num_platforms)){ //todo better validation
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
      std::vector<cl::Device> sel_devices;
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
      }
      else{
        //batch
         num_selected = 1;
         sel_devices.push_back(*cl::platforms[selectedPlat].devices[selectedDev-2]);
         selectedDev = 1;
      }
      if (selectedDev == 1) {
        if (num_selected == 0) {
          // use defaults
        } else {
          // load selected
          cl_context context;
          cl_command_queue cmd_queue;
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
      cout << DASH50 << CLEARN "Starting, Type 0 to Quit \n" << DASH50 << std::endl;
      exps[selectedExp]->Start(1000);
      st = WORK;
      break;
    case WORK:
      if (exps[selectedExp]->IsRunning()) {
          if (!batch) {
          // bare in mind that the program will block here.
          int a = nopromptValidated<int, int>([](int j) { return (j >= 0); });
          if (a == 0) {
            cout << "Stopping" << std::endl;
            exps[selectedExp]->Stop();
          }
        }
      } else {
        st = LOADOUT;
      }
      break;
    }
  }
  for (auto e : exps) {
    delete e;
  }

  std::cout << "\nbye!\n";
}