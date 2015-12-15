#pragma once
#include "experiment.h"

class Exp_Cpu_BitReducer : public Experiment {
public:
	Exp_Cpu_BitReducer();
	~Exp_Cpu_BitReducer();
	void Shutdown();

	void Init2(bool batch, int selectedPlat, std::vector<int> selectedDevices);

private:
	void Start(unsigned int num_runs, const std::vector<int> options);
};
