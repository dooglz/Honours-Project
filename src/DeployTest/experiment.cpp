#include "experiment.h"
#include <thread>
#include <fstream>
#include <iostream>
using namespace std;

Experiment::Experiment(const unsigned int minCu, const unsigned int maxCU, const std::string &name,
                       const std::string &description)
    : minCu(minCu), maxCU(maxCU), name(name), description(description) {
  running = false;
  should_run = false;
}

Experiment::~Experiment() { Shutdown(); }

void Experiment::Shutdown() {
  running = false;
  should_run = false;
};

void Experiment::Stop() {
  printf("stopping thread\n");
  should_run = false;
};

bool Experiment::IsRunning() { return running; }

bool Experiment::ShouldRun() { return should_run; }