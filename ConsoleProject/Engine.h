#pragma once
#include "PrintMachine.h"

class Engine {
public:
	Engine();
	virtual ~Engine() = default;

	void start();
	void run();
	void cleanup();
private:
	PrintMachine* m_printer;
};