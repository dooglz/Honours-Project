#include <stdint.h>
#include <vector>
#include "opencl_utils.h"

class Experiment {
public:
  Experiment();
  ~Experiment();
  virtual uint16_t GetMinCu() = 0;
  virtual uint16_t getMax() = 0;
  virtual void init(cl_context &context, std::vector<cl::device> &devices, cl::platform platform);
  virtual void shutdown();
  virtual void start(uint16_t num_runs);

private:
};
