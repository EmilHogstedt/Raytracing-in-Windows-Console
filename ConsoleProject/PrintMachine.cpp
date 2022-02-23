#include "pch.h"
#include "PrintMachine.h"

PrintMachine* PrintMachine::pInstance{ nullptr };
std::vector<std::vector<char>> PrintMachine::m_2DPrintArray;
std::mutex PrintMachine::m_Lock;
int PrintMachine::m_fps = 60;
size_t PrintMachine::currentWidth = 0;
size_t PrintMachine::currentHeight = 0;
bool PrintMachine::m_running = true;
char* PrintMachine::m_printBuffer = nullptr;
HANDLE PrintMachine::m_handle;

//Sets up the consolemode. Rename this.
bool DisableConsoleQuickEdit(HANDLE consoleHandle) {
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
		PrintMachine::GetInstance()->SetRunning(false);
		break;
	case CTRL_LOGOFF_EVENT:
		PrintMachine::GetInstance()->SetRunning(false);
		break;
	case CTRL_SHUTDOWN_EVENT:
		PrintMachine::GetInstance()->SetRunning(false);
		break;
	}
	return TRUE;
}

void PrintMachine::SetRunning(bool state)
{
	m_running = state;
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
	delete m_printBuffer;
	delete pInstance;
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
	m_fps = fps;
}

std::vector<std::vector<char>>* PrintMachine::Get2DArray()
{
	return &m_2DPrintArray;
}

void PrintMachine::SendData(size_t x, size_t y, char pixelData)
{
	m_2DPrintArray[y][x] = pixelData;
}

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
	//Clear the console before printing.
	ClearConsole();

	memset(m_printBuffer, 0, sizeof(m_printBuffer));
	for (size_t i = 0; i < currentHeight; i++)
	{
		for (size_t j = 0; j < m_2DPrintArray[i].size(); j++)
		{
			m_printBuffer[j + i * (m_2DPrintArray[i].size() + 1)] = m_2DPrintArray[i][j];
		}
		m_printBuffer[m_2DPrintArray[i].size() * (i + 1) + i] = '\n';
	}

	fwrite(m_printBuffer, sizeof(char), currentHeight * (currentWidth + 1), stdout);
	std::cout << "FPS: " << m_fps << "         \n";
	printf("\x1b[31mThis text has a red foreground using SGR.31.\r\n");
	printf("\x1b[mThis text has returned to default colors using SGR.0 implicitly.\r\n");

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