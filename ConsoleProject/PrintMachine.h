#pragma once
#include "Timer.h"

#define WIDTHLIMIT 1000
#define HEIGHTLIMIT 500

class PrintMachine {
private:
	static PrintMachine* pInstance;
	static std::mutex m_Lock;

protected:
	PrintMachine(size_t, size_t);
	~PrintMachine() = default;

public:
	PrintMachine(PrintMachine& other) = delete;
	void operator=(const PrintMachine&) = delete;

public:
	//Called at the beginning of the program to create the instance if it is not already created.
	//This is done in this function instead of in GetInstance to reduce wait time for threads.
	static void CreatePrintMachine(size_t, size_t);
	//Returns the instance if there is one.
	static PrintMachine* GetInstance();
	static bool CheckIfRunning();
	static void TerminateThread();

	//Before closing program
	static void CleanUp();

public:
	static const bool ChangeSize(size_t, size_t);
	static void SendData(size_t, size_t, char);
	static std::vector<std::vector<char>>* Get2DArray();

	static void Fill(char);
	static const bool Print();
	static void UpdateFPS(int);

	static size_t* GetBackBufferSwap();
	static std::mutex* GetBackBufferMutex();
	static char* GetBackBuffer();
	static char* GetDeviceBackBuffer();
	static size_t GetWidth();
	static size_t GetHeight();
	static HANDLE GetConsoleHandle();

	static std::vector<std::vector<char>> m_2DPrintArray;
private:
	static HANDLE m_handle;

	static void ClearConsole();

	static int m_renderingFps;
	static int m_printingFps;
	static Time* m_timer;
	static int m_printingFpsCounter;
	static float m_printingFpsTimer;

	static size_t currentWidth;
	static size_t currentHeight;

	static bool m_running;
	static bool m_terminateThread;

	static char* m_printBuffer;
	static char* m_backBuffer;
	static char* m_deviceBackBuffer;
	
	static std::thread m_printThread;
	static std::mutex m_backBufferMutex;
	static size_t m_backBufferSwap;
};
