#pragma once
#include "pch.h"

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

	//Before closing program
	static void CleanUp();

public:
	static const bool ChangeSize(size_t, size_t);
	static void SendData(size_t, size_t, char);
	static std::vector<std::vector<char>>* Get2DArray();

	static void Fill(char);
	static const bool Print();
	static void UpdateFPS(int);

	static size_t GetWidth();
	static size_t GetHeight();

	static std::vector<std::vector<char>> m_2DPrintArray;
private:
	static void ClearConsole();

	static int m_fps;

	static size_t currentWidth;
	static size_t currentHeight;
};

PrintMachine* PrintMachine::pInstance{ nullptr };
std::vector<std::vector<char>> PrintMachine::m_2DPrintArray;
std::mutex PrintMachine::m_Lock;
int PrintMachine::m_fps = 60;
size_t PrintMachine::currentWidth = 0;
size_t PrintMachine::currentHeight = 0;

//--------------------------------------------------------

PrintMachine::PrintMachine(size_t x, size_t y)
{
	currentWidth = x;
	currentHeight = y;
	m_2DPrintArray.resize(y);
	for (size_t i = 0; i < m_2DPrintArray.size(); i++)
		m_2DPrintArray[i].resize(x);
}

void PrintMachine::CreatePrintMachine(size_t sizeX = 0, size_t sizeY = 0)
{
	if (!pInstance)
		pInstance = DBG_NEW PrintMachine(sizeX, sizeY);
}

PrintMachine* PrintMachine::GetInstance()
{
	return pInstance;
}

void PrintMachine::CleanUp()
{
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
	//char emptyBuffer[100];
	//memset(emptyBuffer, '\n', sizeof(emptyBuffer));
	//fwrite(emptyBuffer, sizeof(emptyBuffer), 1, stdout);

	char buffer[WIDTHLIMIT * HEIGHTLIMIT + HEIGHTLIMIT];
	memset(buffer, 0, sizeof(buffer));
	for (size_t i = 0; i < m_2DPrintArray.size(); i++)
	{
		for (size_t j = 0; j < m_2DPrintArray[i].size(); j++)
		{
			buffer[j + i * (m_2DPrintArray[i].size() + 1)] = m_2DPrintArray[i][j];
		}
		buffer[m_2DPrintArray[i].size() * (i + 1) + i] = '\n';
	}
	
	fwrite(buffer, sizeof(char), m_2DPrintArray.size() * (m_2DPrintArray[0].size() + 1), stdout);
	std::cout << "FPS: " << m_fps << "         \n";
	return true;
}

void PrintMachine::ClearConsole() {
	HANDLE                     hStdOut;
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	DWORD                      count;
	DWORD                      cellCount;
	COORD                      homeCoords = { 0, 0 };

	hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
	if (hStdOut == INVALID_HANDLE_VALUE) return;

	/* Get the number of cells in the current buffer */
	if (!GetConsoleScreenBufferInfo(hStdOut, &csbi)) return;
	cellCount = csbi.dwSize.X * csbi.dwSize.Y;

	/* Fill the entire buffer with spaces */
	/*
	if (!FillConsoleOutputCharacter(
		hStdOut,
		(TCHAR)' ',
		cellCount,
		homeCoords,
		&count
	)) return;
	
	/* Fill the entire buffer with the current colors and attributes */
	
	if (!FillConsoleOutputAttribute(
		hStdOut,
		csbi.wAttributes,
		2000,
		homeCoords,
		&count
	)) return;
	
	/* Move the cursor home */
	SetConsoleCursorPosition(hStdOut, homeCoords);
}