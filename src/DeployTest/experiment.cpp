#include "experiment.h"
#include <thread>

Experiment::Experiment(const uint16_t minCu, const uint16_t maxCU, const std::string& name,
                       const std::string& description)
    : minCu(minCu), maxCU(maxCU), name(name), description(description) {
  running = false;
  should_run = false;
}

Experiment::~Experiment() {
  if (IsRunning()){
    Stop();
  }
}

void Experiment::Init(cl_context &context, std::vector<cl::device> &devices,
                      cl::platform platform){};

void Experiment::Shutdown(){
  if (IsRunning()){
    Stop();
  }
  running = false;
  should_run = false;
};

void Experiment::Start(uint16_t num_runs) {
  printf("start\n");
  if (IsRunning()) {
    printf("Experiment Already Running\n");
    return;
  }

  printf("Starting Experiment\n");
  {
    std::lock_guard<std::mutex> lock(should_run_mutex);
    should_run = true;
  }

  // create the Experiment Thread
  std::thread workThread(&Experiment::Work, this, num_runs);
  workThread.detach();

  // check that everything went ok before returning.
  uint8_t timeout = 0;
  while (!IsRunning()) {
    // spin
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ++timeout;
    if (timeout >= 255) {
      printf("Experiment creation timeout\n");
      Stop();
      return;
    }
  }
};

void Experiment::Stop() {
  printf("stopping thread\n");
  {
    std::lock_guard<std::mutex> lock(should_run_mutex);
    should_run = false;
  }
  uint8_t timeout = 0;
  while (IsRunning()) {
    // spin
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ++timeout;
    if (timeout >= 255) {
      printf("Experiment Stop timeout\n");
      // TODO: forceQuit
      return;
    }
  }
  printf("stopped\n");
};

bool Experiment::IsRunning() {
  std::lock_guard<std::mutex> lock(running_mutex);
  return running;
}

bool Experiment::ShouldRun() {
  std::lock_guard<std::mutex> lock(should_run_mutex);
  return should_run;
}
