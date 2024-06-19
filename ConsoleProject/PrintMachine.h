#pragma once
#include "Timer.h"

#define WIDTHLIMIT 1000
#define HEIGHTLIMIT 500

class PrintMachine {
private:
	static std::mutex m_Lock;

protected:
	PrintMachine() = delete;

public:
	enum PrintMode { ASCII = 0, PIXEL, RGB_ASCII, RGB_PIXEL, RGB_NORMALS };

	static void Start(const size_t x, const size_t y);
	static void CleanUp();

	static bool CheckIfRunning();
	static void SetDebugInfo(const std::string& debugString);
	static void TerminateThread();

public:
	//Thread function.
	static bool Print();

	static void UpdateRenderingFPS(const int fps);

	static bool ChangeSize(const size_t x, const size_t y);

	static std::mutex* GetBackBufferMutex();
	static char* GetBackBuffer();
	static char DEVICE_MEMORY_PTR GetDeviceBackBuffer();
	static void ResetDeviceBackBuffer();
	static size_t GetWidth();
	static size_t GetHeight();
	static size_t GetMaxSize();
	static HANDLE GetConsoleInputHandle();
	static HANDLE GetConsoleOutputHandle();
	static size_t GetPrintSize();
	static PrintMode GetPrintMode();

	static void ResetBackBuffer();
	static void FlagForBufferSwap();
	static void SetPrintSize(const size_t newSize);
	static void SetPrintMode(const PrintMode mode);

private:
	static HANDLE m_inputHandle;
	static HANDLE m_outputHandle;

	static PrintMode m_printMode;

	static void ResetConsolePointer();

	static int m_renderingFps;
	static int m_printingFps;

	static std::unique_ptr<Time> m_timer;
	static int m_printingFpsCounter;
	static float m_printingFpsTimer;

	static size_t currentWidth;
	static size_t currentHeight;
	static size_t m_maxSize;

	static bool m_running;
	static bool m_terminateThread;

	//static char* m_printBuffer;
	static std::unique_ptr<char[]> m_printBuffer;

	//These are two different buffers.
	//The first is a buffer that is written to on the CPU using the minimized result of the GPU processing, and then swapped with the printbuffer.
	//static char* m_backBuffer;
	static std::unique_ptr<char[]> m_backBuffer;

	//The second is the buffer which is used to write to from the GPU threads.
	static char DEVICE_MEMORY_PTR m_deviceBackBuffer;

	static size_t m_printSize;
	static size_t m_backBufferPrintSize;
	static std::string m_debugInfo;

	static std::thread m_printThread;
	static std::mutex m_backBufferMutex;
	static bool m_bShouldSwapBuffer;

	//20 characters per pixel allows RGB colors.
	static const size_t m_charsPerPixel = 20;
};
