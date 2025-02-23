#include "pch.h"
#include "PrintMachine.h"

#include "Timer.h"

std::mutex PrintMachine::m_Lock;

int PrintMachine::m_renderingFps = 60;
int PrintMachine::m_printingFps = 60;
std::unique_ptr<Time> PrintMachine::m_timer = nullptr;
int PrintMachine::m_printingFpsCounter = 0;
float PrintMachine::m_printingFpsTimer = 0.0f;

size_t PrintMachine::currentWidth = 0;
size_t PrintMachine::currentHeight = 0;
size_t PrintMachine::m_maxSize = 0;

bool PrintMachine::m_running = true;
bool PrintMachine::m_terminateThread = false;

std::unique_ptr<char[]> PrintMachine::m_printBuffer = nullptr;
std::unique_ptr<char[]> PrintMachine::m_backBuffer = nullptr;
size_t PrintMachine::m_printSize = 0;
size_t PrintMachine::m_backBufferPrintSize = 0;
std::string PrintMachine::m_debugInfo = "";

HANDLE PrintMachine::m_inputHandle;
HANDLE PrintMachine::m_outputHandle;

std::thread PrintMachine::m_printThread;
std::mutex PrintMachine::m_backBufferMutex;
bool PrintMachine::m_bShouldSwapBuffer = false;


//Sets up the consolemode. Rename this.
bool ConfigureConsoleInputMode(HANDLE consoleHandle)
{
	//Get current console mode
	unsigned int consoleMode = 0;
	if (!GetConsoleMode(consoleHandle, (LPDWORD)&consoleMode)) {
		//ERROR: Unable to get console mode.
		return false;
	}

	// Clear the quick edit bit in the mode flags
	consoleMode &= ~ENABLE_QUICK_EDIT_MODE;
	consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
	consoleMode |= ENABLE_MOUSE_INPUT;

	//Set the new mode
	if (!SetConsoleMode(consoleHandle, consoleMode)) {
		//ERROR: Unable to set console mode
		return false;
	}

	return true;
}

bool ConfigureConsoleOutputMode(HANDLE outputHandle)
{
	//Get current console mode
	unsigned int consoleMode = 0;
	if (!GetConsoleMode(outputHandle, (LPDWORD)&consoleMode)) {
		//ERROR: Unable to get console mode.
		return false;
	}

	consoleMode |= ENABLE_PROCESSED_OUTPUT;
	consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;

	//Set the new mode
	if (!SetConsoleMode(outputHandle, consoleMode)) {
		//ERROR: Unable to set console mode
		return false;
	}

	return true;
}

//Handles console events, does the cleanup when console window is closed.
BOOL WINAPI ConsoleHandler(DWORD CEvent)
{
	switch (CEvent)
	{
	case CTRL_CLOSE_EVENT:
		PrintMachine::TerminateThread();
		break;
	case CTRL_LOGOFF_EVENT:
		PrintMachine::TerminateThread();
		break;
	case CTRL_SHUTDOWN_EVENT:
		PrintMachine::TerminateThread();
		break;
	}
	return TRUE;
}

void PrintMachine::TerminateThread()
{
	m_terminateThread = true;
}

bool PrintMachine::CheckIfRunning()
{
	return m_running;
}

void PrintMachine::Start(const size_t x, const size_t y)
{
	//Set up the console.
	{
		m_outputHandle = GetStdHandle(STD_OUTPUT_HANDLE);
		m_inputHandle = GetStdHandle(STD_INPUT_HANDLE);

		//When this is set to true it guarantees that all c-stream and c++-stream output functions sync correctly.
		//PS: Earlier I had this set to false for some reason? It said that it can increase performance if only using c++ fuinctions, so maybe that is why.
		std::ios_base::sync_with_stdio(true);

		//This command is run at program startup automatically, why have it here?
		//setlocale(LC_ALL, "C");

		printf("\x1b[?25l"); // Disables blinking cursor.

		printf("\x1b]0;Avero's Console RayTracing Engine\x1b\x5c"); //Set the title of the program.

		ConfigureConsoleInputMode(m_inputHandle); //Disables being able to click in the console window.
		ConfigureConsoleOutputMode(m_outputHandle); //Enables processed output and virtual terminal processing.

		//The console handler handles when the console is closed for any reason.
		if (!SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleHandler, TRUE))
		{
			assert(false && "Unable to install handler!");
		}
	}
	
	currentWidth = x;
	currentHeight = y;

	//The console window is width * characters per pixel * height.
	m_maxSize = (m_charsPerPixel * currentWidth * currentHeight) + currentHeight;

	m_printBuffer = std::make_unique<char[]>(m_maxSize);

	m_backBuffer = std::make_unique<char[]>(m_maxSize);

	m_printSize = m_maxSize;

	m_timer = std::make_unique<Time>();

	m_printThread = std::thread(&Print);
	m_printThread.detach();
}

void PrintMachine::CleanUp()
{
	ResetConsolePointer();

	//Reset color.
	printf("\x1b[m");

	//Soft reset settings.
	printf("\x1b[!p");

	//Clear the screen.
	printf("\x1b[2J");
}

const std::mutex* PrintMachine::GetBackBufferMutex()
{
	return &m_backBufferMutex;
}

const char* PrintMachine::GetBackBuffer()
{
	return m_backBuffer.get();
}

void PrintMachine::SetDataInBackBuffer(const char* data, const size_t size)
{
	m_backBufferMutex.lock();

	//Copy the data over to the backbuffer.
	memcpy(m_backBuffer.get(), data, size);

	//Signal the printmachine that it should swap buffers.
	PrintMachine::FlagForBufferSwap();

	//Change print size to reflect the data being printed.
	PrintMachine::SetPrintSize(size);

	m_backBufferMutex.unlock();
}

const size_t PrintMachine::GetWidth()
{
	return currentWidth;
}

const size_t PrintMachine::GetHeight()
{
	return currentHeight;
}

const size_t PrintMachine::GetMaxSize()
{
	return m_maxSize;
}

const HANDLE PrintMachine::GetConsoleInputHandle()
{
	return m_inputHandle;
}

const HANDLE PrintMachine::GetConsoleOutputHandle()
{
	return m_outputHandle;
}

const size_t PrintMachine::GetPrintSize()
{
	return m_printSize;
}

bool PrintMachine::ChangeSize(const size_t x, const size_t y)
{
	currentWidth = x;
	currentHeight = y;

	return true;
}

void PrintMachine::UpdateRenderingFPS(const int fps)
{
	m_renderingFps = fps;
}

void PrintMachine::SetDebugInfo(const std::string& debugString)
{
	m_debugInfo = debugString;
}

void PrintMachine::FlagForBufferSwap()
{
	m_bShouldSwapBuffer = true;
}

void PrintMachine::SetPrintSize(const size_t newSize)
{
	m_backBufferPrintSize = newSize;
}

void PrintMachine::ResetBackBuffer()
{
	memset(m_backBuffer.get(), 0, sizeof(char) * m_maxSize);
}

bool PrintMachine::Print()
{
	while (!m_terminateThread)
	{
		m_timer->Update();
		m_printingFpsCounter++;

		m_printingFpsTimer += m_timer->DeltaTime();

		//Once every second we update the fps.
		if (m_printingFpsTimer >= 1.0f)
		{
			m_printingFps = m_printingFpsCounter;
			m_printingFpsTimer = 0.0f;
			m_printingFpsCounter = 0;
		}

		//Check if we need to swap to the backbuffer.
		m_backBufferMutex.lock();
		if (m_bShouldSwapBuffer)
		{
			m_bShouldSwapBuffer = false;

			m_printSize = m_backBufferPrintSize;

			//Which one of these are faster?
			//memcpy(m_printBuffer.get(), m_backBuffer.get(), m_printSize);
			m_printBuffer.swap(m_backBuffer);
		}
		m_backBufferMutex.unlock();

		//Reset pointer and print the data.
		ResetConsolePointer();
		fwrite(m_printBuffer.get(), 1, m_printSize, stdout);
		
		//DWORD written;
		//WriteConsoleOutputCharacterA(m_outputHandle, m_printBuffer, m_printSize, {0, 0}, &written);
		//WriteConsoleA(m_outputHandle, m_printBuffer, m_printSize, &written, NULL);
		//std::cout.write(m_printBuffer.get(), m_printSize);
		//printf("%.*s", (unsigned int)(37 * currentHeight * currentWidth + currentHeight), m_printBuffer);
		printf("\x1b[m");
		printf("Rendering FPS: %d    \n", m_renderingFps);
		printf("Printing FPS: %d    \n", m_printingFps);
		//DEBUG ONLY
		//std::cout << m_debugInfo << std::endl;
	}
	m_running = false;
	return true;
}

void PrintMachine::ResetConsolePointer()
{
	SetConsoleCursorPosition(m_outputHandle, {0, 0});
}