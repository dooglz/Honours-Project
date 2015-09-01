#include "experiment.h"

void Experiment::init(cl_context &context, std::vector<cl::device> &devices,
                      cl::platform platform){};
void Experiment::shutdown(){};
void Experiment::start(uint16_t num_runs){};

Experiment::Experiment() {}

Experiment::~Experiment() {}

Experiment::~Experiment() {}