#pragma once
#include "pch.h"
#include "PrintMachine.h"
#include "Timer.h"
#include "Camera.h"
#include "Scene.h"

void calculatePixel()
{


}

bool DisableConsoleQuickEdit() {
	const unsigned int ENABLE_QUICK_EDIT = 0x0040;
	HANDLE consoleHandle = GetStdHandle(STD_INPUT_HANDLE);

	// get current console mode
	unsigned int consoleMode = 0;
	if (!GetConsoleMode(consoleHandle, (LPDWORD)&consoleMode)) {
		// ERROR: Unable to get console mode.
		return false;
	}

	// Clear the quick edit bit in the mode flags
	consoleMode &= ~ENABLE_QUICK_EDIT;

	// set the new mode
	if (!SetConsoleMode(consoleHandle, consoleMode)) {
		// ERROR: Unable to set console mode
		return false;
	}

	return true;
}

//RayMarching Singleton Engine.
class Engine {
	static Engine* pInstance;
protected:
	Engine();
	~Engine();
public:
	Engine(Engine& other) = delete;
	void operator=(const Engine&) = delete;

	//Called at the beginning of the program to create the instance if it is not already created.
	//This is done in this function instead of in GetInstance to reduce wait time for threads.
	static void CreateEngine();
	//Returns the instance if there is one.
	static Engine* GetInstance();
	static void Start();
	//Is called in our game loop.
	static bool Run();
	static void CleanUp();
private:
	static void Render();

	static Time* m_timer;
	static Camera* m_camera;
	static Scene* m_scene;
	static long double m_frameTimer;
	static long double m_fpsTimer;
	static int m_fps;
};

Engine* Engine::pInstance{ nullptr };
Time* Engine::m_timer{ nullptr };
Camera* Engine::m_camera{ nullptr };
Scene* Engine::m_scene{ nullptr };
long double Engine::m_frameTimer = 0.0f;
long double Engine::m_fpsTimer = 0.0f;
int Engine::m_fps = 0;
//----------------------------------------

void Engine::CreateEngine()
{
	if (!pInstance)
		pInstance = DBG_NEW Engine();
}

Engine* Engine::GetInstance()
{
	return pInstance;
}

Engine::Engine() {
	//Console stuff
	DisableConsoleQuickEdit();
	std::ios::sync_with_stdio(false);

	//Engine objects
	m_timer = DBG_NEW Time();
	m_camera = DBG_NEW Camera();
	m_scene = DBG_NEW Scene();
}

Engine::~Engine()
{
	delete m_timer;
	delete m_camera;
	delete m_scene;
}

//Handles console events, does the cleanup when console window is closed.
BOOL WINAPI ConsoleHandler(DWORD CEvent)
{
	switch (CEvent)
	{
	case CTRL_CLOSE_EVENT:
		Engine::GetInstance()->CleanUp();
		break;
	case CTRL_LOGOFF_EVENT:
		Engine::GetInstance()->CleanUp();
		break;
	case CTRL_SHUTDOWN_EVENT:
		Engine::GetInstance()->CleanUp();
		break;
	}
	return TRUE;
}

void Engine::Start()
{
	if (!SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleHandler, TRUE))
	{
		// unable to install handler... 
		// display message to the user
		printf("Unable to install handler!\n");
		assert(false);
	}
	PrintMachine::CreatePrintMachine(50, 20);
	PrintMachine::GetInstance()->Fill(' '); //Temp
	m_camera->Init();
}

bool Engine::Run()
{
	m_timer->Update();
	m_fps++;

	m_frameTimer += m_timer->DeltaTime();
	m_fpsTimer += m_timer->DeltaTime();

	//Render
	Render();



	//Once every second we update the fps.
	if (m_fpsTimer >= 1.0f)
	{
		PrintMachine::GetInstance()->UpdateFPS(m_fps);
		m_fpsTimer = 0.0f;
		m_fps = 0;
	}
	//VSYNC
	//If its bigger than 60fps we print
	//if (m_frameTimer >= 1.0f / 144.0f)
	//{
		PrintMachine::GetInstance()->Print();
		//m_frameTimer = 0.0f;
	//}
	
	return true;
}

void Engine::Render()
{
	m_camera->Update();

	//Like our pixel shader.
	std::vector<std::thread> pixels;
	for (size_t i = 0; i < PrintMachine::GetInstance()->GetHeight(); i++)
	{
		for (size_t j = 0; j < PrintMachine::GetInstance()->GetWidth(); j++)
		{
			pixels.push_back(std::thread(
				calculatePixel,
				//Arguments to pixel shader.
				m_camera->GetVMatrix(), //Need the inverse here instead
				m_camera->GetPMatrix(),
				m_camera->GetPos(),
				std::ref()

			));
		}
	}
	for (size_t i = 0; i < pixels.size(); i++)
	{
		pixels[i].join();
	}
}

void Engine::CleanUp()
{
	PrintMachine::GetInstance()->CleanUp();
	delete pInstance;
}

