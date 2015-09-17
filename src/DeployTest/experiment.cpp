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

Experiment::~Experiment() {
  if (IsRunning()) {
    Stop();
  }
}
void Experiment::Init(cl_context &context, cl_command_queue &cq, std::vector<cl::device> &devices, cl::platform platform){};

void Experiment::Shutdown() {
  if (IsRunning()) {
    Stop();
  }
  running = false;
  should_run = false;
};

void Experiment::Start(unsigned int num_runs) {
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
  unsigned int timeout = 0;
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
    lock_guard<mutex> lock(should_run_mutex);
    should_run = false;
  }
  unsigned int timeout = 0;
  while (IsRunning()) {
    // spin
    this_thread::sleep_for(std::chrono::milliseconds(10));
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
  lock_guard<mutex> lock(running_mutex);
  return running;
}

bool Experiment::ShouldRun() {
  lock_guard<mutex> lock(should_run_mutex);
  return should_run;
}

const void Experiment::PrintToCSV(const string &collumn1, const string &collumn2,
                                  const vector<Timer> &times, const string &filename) {
  ofstream data(filename + ".csv", ofstream::out);
  data << collumn1 << "," << collumn2 << endl;
  for (auto d : times) {
    data << d.name << "," << chrono::duration_cast<chrono::nanoseconds>(d.Duration()).count()
         << endl;
  }
  data.close();
  cout << "printed to" << filename << ".csv" << endl;
}

const void Experiment::PrintToCSV(const vector<vector<string>> v, const string &filename) {
  ofstream data(filename + ".csv", ofstream::out);
  bool b = true;
  unsigned int i = 0;
  while (b) {
    b = false;
    for (auto vec : v) {
      if (i <= vec.size()) {
        b = true;
        data << vec[i] << ",";
      }
    }
    data << endl;
    ++i;
  }
  data.close();
  cout << "printed to" << filename << ".csv" << endl;
}
