#include "pch.h"
#pragma once

#define WIDTHLIMIT 100
#define HEIGHTLIMIT 50

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
	static void Fill(char);
	static const bool Print();

private:
	static void ClearConsole();

	static std::vector<std::vector<char>> m_2DPrintArray;
};

PrintMachine* PrintMachine::pInstance{ nullptr };
std::vector<std::vector<char>> PrintMachine::m_2DPrintArray;
std::mutex PrintMachine::m_Lock;

//--------------------------------------------------------

PrintMachine::PrintMachine(size_t x, size_t y)
{
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

const bool PrintMachine::ChangeSize(size_t x, size_t y)
{
	if (x > WIDTHLIMIT)
		return false;
	if (y > HEIGHTLIMIT)
		return false;

	m_2DPrintArray = {};
	m_2DPrintArray.resize(x);
	for (size_t i = 0; i < m_2DPrintArray.size(); i++)
		m_2DPrintArray.resize(y);

	return true;
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
	//ClearConsole();
	system("cls");

	//Print
	char n = '\n';
	for (size_t i = 0; i < m_2DPrintArray.size(); i++)
	{
		//for (size_t j = 0; j < m_2DPrintArray[i].size(); j++)
		fwrite(m_2DPrintArray[i].data(), sizeof(char), m_2DPrintArray[i].size(), stdout);
			//std::cout << m_2DPrintArray[i][j];
		fwrite(&n, sizeof(char), 1, stdout);
	}
		
	return true;
}

void PrintMachine::ClearConsole() {
	COORD topLeft = { 0, 0 };
	HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_SCREEN_BUFFER_INFO screen;
	DWORD written;

	GetConsoleScreenBufferInfo(console, &screen);
	FillConsoleOutputCharacterA(
		console, ' ', screen.dwSize.X * screen.dwSize.Y, topLeft, &written
	);
	FillConsoleOutputAttribute(
		console, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE,
		screen.dwSize.X * screen.dwSize.Y, topLeft, &written
	);
	SetConsoleCursorPosition(console, topLeft);
}