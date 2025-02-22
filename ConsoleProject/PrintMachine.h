#pragma once

#define WIDTHLIMIT 1000
#define HEIGHTLIMIT 500

class Time;

class PrintMachine {
private:
	static std::mutex m_Lock;

protected:
	PrintMachine() = delete;

public:
	static void Start(const size_t x, const size_t y);
	static void CleanUp();

	static bool CheckIfRunning();
	static void SetDebugInfo(const std::string& debugString);
	static void TerminateThread();

public:
	//This function is run on a seperate thread.
	static bool Print();

	static void UpdateRenderingFPS(const int fps);

	static bool ChangeSize(const size_t x, const size_t y);

	static const std::mutex* GetBackBufferMutex();

	static const char* GetBackBuffer();
	static void SetDataInBackBuffer(const char* data, const size_t size);

	static const size_t GetWidth();
	static const size_t GetHeight();
	static const size_t GetMaxSize();
	static const HANDLE GetConsoleInputHandle();
	static const HANDLE GetConsoleOutputHandle();
	static const size_t GetPrintSize();

	static void ResetBackBuffer();
	static void FlagForBufferSwap();
	static void SetPrintSize(const size_t newSize);

private:
	static HANDLE m_inputHandle;
	static HANDLE m_outputHandle;

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

	static std::unique_ptr<char[]> m_printBuffer;

	//A buffer that is written to on the CPU using the minimized result of the GPU processing, and then swapped with the printbuffer.
	static std::unique_ptr<char[]> m_backBuffer;

	static size_t m_printSize;
	static size_t m_backBufferPrintSize;
	static std::string m_debugInfo;

	static std::thread m_printThread;
	static std::mutex m_backBufferMutex;
	static bool m_bShouldSwapBuffer;

	//20 characters per pixel allows RGB colors.
	static const size_t m_charsPerPixel = 20;
};
