#include "openclExp.h"
#include "utils.h"
#include <thread>
#include <fstream>
#include <iostream>
using namespace std;

OpenCLExperiment::OpenCLExperiment(const unsigned int minCu, const unsigned int maxCU,
                                   const std::string &name, const std::string &description)
    : Experiment(minCu, maxCU, name, description) {
  running = false;
  should_run = false;
}

void OpenCLExperiment::Init(cl_context &context, std::vector<cl_command_queue> &cq,
                            std::vector<cl::CLDevice> &devices, cl::Platform platform){};

void OpenCLExperiment::Init2(bool batch) {
  unsigned int selectedPlat = 0;
  if (!batch) {
    cout << "Experiment requires a Minimum number of " << minCu << " devices, and a maximum of "
         << maxCU << std::endl;
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
	selectedPlat = selectedPlat - 2;
    unsigned int selectedDev = 0;
    unsigned int num_selected = 0;
    std::vector<cl::CLDevice> sel_devices;

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
        sel_devices.push_back(*cl::platforms[selectedPlat].devices[i]);
      }
    }
    delete[] selected;

    if (selectedDev == 1) {
      if (num_selected == 0) {
        // use defaults
      } else {
        // load selected
        cl_context context;
        std::vector<cl_command_queue> cmd_queue;
        GetContext(sel_devices, context, cmd_queue);
        Init(context, cmd_queue, sel_devices, cl::platforms[selectedPlat]);
      }
    } else {
      // Cancelled out
      return;
    }
  } else {
    /*
    if (selectedExp < 0 ||
            selectedExp >(2 + cl::total_num_platforms)) { // todo better validation
            cout << "Invalid Platform" << std::endl;
            return 1;
    }
    */
  }
  /*
  if (selectedPlat == 0) {
          st = CHOOSE;
          break;
  }
  selectedPlat = selectedPlat - 2;
  st = CHOOSED;
  */
};
