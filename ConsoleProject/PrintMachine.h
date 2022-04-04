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
	enum PrintMode { ASCII = 0, PIXEL, RGB_ASCII, RGB_PIXEL };
	//Called at the beginning of the program to create the instance if it is not already created.
	//This is done in this function instead of in GetInstance to reduce wait time for threads.
	static void CreatePrintMachine(size_t, size_t);
	//Returns the instance if there is one.
	static PrintMachine* GetInstance();
	static bool CheckIfRunning();
	static void SetDebugInfo(std::string);
	static void TerminateThread();

	//Before closing program
	static void CleanUp();

public:
	static const bool ChangeSize(size_t, size_t);

	static void Fill(char);
	static const bool Print();
	static void UpdateFPS(int);

	static size_t* GetBackBufferSwap();
	static std::mutex* GetBackBufferMutex();
	static char* GetBackBuffer();
	static char* GetDeviceBackBuffer();
	static void ResetDeviceBackBuffer();
	static size_t GetWidth();
	static size_t GetHeight();
	static size_t GetMaxSize();
	static HANDLE GetConsoleHandle();
	static size_t GetPrintSize();
	static PrintMode GetPrintMode();

	static void ResetBackBuffer();
	static void SetBufferSwap(size_t);
	static void SetPrintSize(size_t);
	static void SetPrintMode(PrintMode);
private:
	static HANDLE m_handle;

	static PrintMode m_printMode;

	static void ClearConsole();

	static int m_renderingFps;
	static int m_printingFps;
	static Time* m_timer;
	static int m_printingFpsCounter;
	static float m_printingFpsTimer;

	static size_t currentWidth;
	static size_t currentHeight;
	static size_t m_maxSize;

	static bool m_running;
	static bool m_terminateThread;

	static char* m_printBuffer;
	static char* m_backBuffer;
	static char* m_deviceBackBuffer;
	static size_t m_printSize;
	static size_t m_backBufferPrintSize;
	static std::string m_debugInfo;

	static std::thread m_printThread;
	static std::mutex m_backBufferMutex;
	static size_t m_backBufferSwap;
};
