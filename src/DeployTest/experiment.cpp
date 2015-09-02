#include "experiment.h"
#include <thread>


Experiment::Experiment() {
  running = false;
  should_run = false;
}

Experiment::~Experiment() {}

void Experiment::Init(cl_context &context, std::vector<cl::device> &devices,
                      cl::platform platform){};

void Experiment::Shutdown(){};


void Experiment::Start(uint16_t num_runs) {
  printf("start\n");
  //don't run if already running
  if (IsRunning()) {
    printf("already started\n");

    return;
  }
  printf("starting thread\n");
  {
    std::lock_guard<std::mutex> lock(should_run_mutex);
    should_run = true;
  }
  std::thread workThread(&Experiment::Work, this, num_runs);
  workThread.detach();
};

void Experiment::Stop() {
  printf("stopping thread\n");
  {
    std::lock_guard<std::mutex> lock(should_run_mutex);
    should_run = false;
  }
  while (IsRunning()){
    //spin, TODO: timeout 
  }
  printf("stopped\n");
};

bool Experiment::IsRunning(){
  std::lock_guard<std::mutex> lock(running_mutex);
  return running;
}

bool Experiment::ShouldRun(){
  std::lock_guard<std::mutex> lock(should_run_mutex);
  return should_run;
}
