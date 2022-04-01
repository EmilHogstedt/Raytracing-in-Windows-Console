#include "pch.h"
#include "PrintMachine.h"

PrintMachine* PrintMachine::pInstance{ nullptr };
std::vector<std::vector<char>> PrintMachine::m_2DPrintArray;
std::mutex PrintMachine::m_Lock;

int PrintMachine::m_renderingFps = 60;
int PrintMachine::m_printingFps = 60;
Time* PrintMachine::m_timer{ nullptr };
int PrintMachine::m_printingFpsCounter = 0;
float PrintMachine::m_printingFpsTimer = 0.0f;

size_t PrintMachine::currentWidth = 0;
size_t PrintMachine::currentHeight = 0;
bool PrintMachine::m_running = true;
bool PrintMachine::m_terminateThread = false;

char* PrintMachine::m_printBuffer = nullptr;
char* PrintMachine::m_backBuffer = nullptr;
char* PrintMachine::m_deviceBackBuffer = nullptr;
HANDLE PrintMachine::m_handle;

std::thread PrintMachine::m_printThread;
std::mutex PrintMachine::m_backBufferMutex;
size_t PrintMachine::m_backBufferSwap = 0;


//Sets up the consolemode. Rename this.
bool DisableConsoleQuickEdit(HANDLE consoleHandle) {
	std::ios_base::sync_with_stdio(false);
	setlocale(LC_ALL, "C");
	printf("\x1b[?25l");

	const unsigned int ENABLE_QUICK_EDIT = 0x0040;
	//HANDLE consoleHandle = GetStdHandle(STD_INPUT_HANDLE);

	//Get current console mode
	unsigned int consoleMode = 0;
	if (!GetConsoleMode(consoleHandle, (LPDWORD)&consoleMode)) {
		//ERROR: Unable to get console mode.
		return false;
	}

	// Clear the quick edit bit in the mode flags
	consoleMode &= ~ENABLE_QUICK_EDIT;
	consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
	consoleMode |= ENABLE_MOUSE_INPUT;

	//Set the new mode
	if (!SetConsoleMode(consoleHandle, consoleMode)) {
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
		PrintMachine::GetInstance()->TerminateThread();
		break;
	case CTRL_LOGOFF_EVENT:
		PrintMachine::GetInstance()->TerminateThread();
		break;
	case CTRL_SHUTDOWN_EVENT:
		PrintMachine::GetInstance()->TerminateThread();
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

PrintMachine::PrintMachine(size_t x, size_t y)
{
	currentWidth = x;
	currentHeight = y;
	m_2DPrintArray.resize(y);
	for (size_t i = 0; i < m_2DPrintArray.size(); i++)
		m_2DPrintArray[i].resize(x);

	//+ heightlimit is for the line-ending characters.
	//Needs to be remade if support for color is added.
	m_printBuffer = DBG_NEW char[WIDTHLIMIT * HEIGHTLIMIT + HEIGHTLIMIT];
	memset(m_printBuffer, 0, sizeof(char) * WIDTHLIMIT * HEIGHTLIMIT + HEIGHTLIMIT);
	m_backBuffer = DBG_NEW char[WIDTHLIMIT * HEIGHTLIMIT + HEIGHTLIMIT];
	memset(m_backBuffer, 0, sizeof(char) * WIDTHLIMIT * HEIGHTLIMIT + HEIGHTLIMIT);
	cudaMalloc(&m_deviceBackBuffer, sizeof(char) * WIDTHLIMIT * HEIGHTLIMIT + HEIGHTLIMIT);
	cudaMemset(m_deviceBackBuffer, 0, sizeof(char) * WIDTHLIMIT * HEIGHTLIMIT + HEIGHTLIMIT);

	m_timer = DBG_NEW Time();

	m_printThread = std::thread(&Print);
	m_printThread.detach();
}

void PrintMachine::CreatePrintMachine(size_t sizeX = 0, size_t sizeY = 0)
{
	m_handle = GetStdHandle(STD_INPUT_HANDLE);
	//Console stuff
	DisableConsoleQuickEdit(m_handle); //Disables being able to click in the console window.
	std::ios::sync_with_stdio(false);

	if (!SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleHandler, TRUE))
	{
		// unable to install handler... 
		// display message to the user
		printf("Unable to install handler!\n");
		assert(false);
	}

	if (!pInstance)
		pInstance = DBG_NEW PrintMachine(sizeX, sizeY);

}

PrintMachine* PrintMachine::GetInstance()
{
	return pInstance;
}

void PrintMachine::CleanUp()
{
	delete m_timer;
	cudaFree(m_deviceBackBuffer);
	delete m_backBuffer;
	delete m_printBuffer;
	delete pInstance;
}

size_t* PrintMachine::GetBackBufferSwap()
{
	return &m_backBufferSwap;
}

std::mutex* PrintMachine::GetBackBufferMutex()
{
	return &m_backBufferMutex;
}

//The backbuffer mutex NEEDS to be locked before calling this function.
char* PrintMachine::GetBackBuffer()
{
	return m_backBuffer;
}

char* PrintMachine::GetDeviceBackBuffer()
{
	return m_deviceBackBuffer;
}

size_t PrintMachine::GetWidth()
{
	return currentWidth;
}

size_t PrintMachine::GetHeight()
{
	return currentHeight;
}

HANDLE PrintMachine::GetConsoleHandle()
{
	return m_handle;
}

const bool PrintMachine::ChangeSize(size_t x, size_t y)
{
	if (x > WIDTHLIMIT)
		return false;
	if (y > HEIGHTLIMIT)
		return false;

	currentWidth = x;
	currentHeight = y;
	m_2DPrintArray = {};
	m_2DPrintArray.resize(x);
	for (size_t i = 0; i < m_2DPrintArray.size(); i++)
		m_2DPrintArray.resize(y);

	return true;
}

void PrintMachine::UpdateFPS(int fps)
{
	m_renderingFps = fps;
}

std::vector<std::vector<char>>* PrintMachine::Get2DArray()
{
	return &m_2DPrintArray;
}
/*
void PrintMachine::SendData(size_t x, size_t y, char pixelData)
{
	m_2DPrintArray[y][x] = pixelData;
}*/

void PrintMachine::Fill(char character)
{
	for (size_t i = 0; i < m_2DPrintArray.size(); i++)
	{
		for (size_t j = 0; j < m_2DPrintArray[i].size(); j++)
			m_2DPrintArray[i][j] = character;
	}
}

const bool PrintMachine::Print()
{
	while (!m_terminateThread)
	{
		m_timer->UpdatePrinting();
		m_printingFpsCounter++;

		m_printingFpsTimer += m_timer->DeltaTimePrinting();

		//Once every second we update the fps.
		if (m_printingFpsTimer >= 1.0f)
		{
			m_printingFps = m_printingFpsCounter;
			m_printingFpsTimer = 0.0f;
			m_printingFpsCounter = 0;
		}

		//Check if we need to swap to the backbuffer.
		m_backBufferMutex.lock();
		if (m_backBufferSwap)
		{
			m_backBufferSwap = 0;
			char* temp = m_printBuffer;
			m_printBuffer = m_backBuffer;
			m_backBuffer = m_printBuffer;
		}
		m_backBufferMutex.unlock();

		//Clear the console and print the data.
		ClearConsole();
		fwrite(m_printBuffer, sizeof(char), currentHeight * (currentWidth + 1), stdout);
		printf("Rendering FPS: %d    \n", m_renderingFps);
		printf("Printing FPS: %d    \n", m_printingFps);
	}
	m_running = false;
	//Clear the console before printing.
	//ClearConsole();

	//memset(m_printBuffer, 0, sizeof(m_printBuffer));
	//gpuErrchk(cudaDeviceSynchronize());
	//cudaMemcpy(m_printBuffer, m_devicePrintBuffer, sizeof(char) * (currentWidth + 1) * currentHeight, cudaMemcpyDeviceToHost);
	
	//fwrite(m_printBuffer, sizeof(char), currentHeight * (currentWidth + 1), stdout);
	//printf("FPS: %d    \n", m_fps);
	// 
	//printf("\x1b[31mThis text has a red foreground using SGR.31.\r\n");
	//printf("\x1b[mThis text has returned to default colors using SGR.0 implicitly.\r\n");

	return true;
}

//Very strange functionality. Does not behave as it should.
void PrintMachine::ClearConsole() {
	/*
	HANDLE                     hStdOut;
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	DWORD                      count;
	DWORD                      cellCount;
	COORD                      homeCoords = { 0, 0 };

	hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
	if (hStdOut == INVALID_HANDLE_VALUE) return;

	//Get the number of cells in the current buffer
	if (!GetConsoleScreenBufferInfo(hStdOut, &csbi)) return;
	cellCount = csbi.dwSize.X * csbi.dwSize.Y;

	//Fill the entire buffer with spaces
	
	if (!FillConsoleOutputCharacter(
		hStdOut,
		(TCHAR)' ',
		cellCount,
		homeCoords,
		&count
	)) return;

	//Fill the entire buffer with the current colors and attributes
	
	if (!FillConsoleOutputAttribute(
		hStdOut,
		csbi.wAttributes,
		2000,
		homeCoords,
		&count
	)) return;

	//Move the cursor home
	SetConsoleCursorPosition(hStdOut, homeCoords);
	*/
	HANDLE hOut;
	COORD Position;

	hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	
	Position.X = 0;
	Position.Y = 0;
	SetConsoleCursorPosition(hOut, Position);
}