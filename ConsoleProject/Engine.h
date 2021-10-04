#pragma once
#include "pch.h"
#include "PrintMachine.h"
#include "Camera.h"
#include "Timer.h"

#include "Scene.h"

void calculatePixel(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object*> culledObjects, size_t objectNr, std::vector<std::vector<char>>* twoDArray, size_t currentWidth, size_t currentHeight, size_t threadWidthPos, size_t threadHeightPos)
{
	//Convert our coordinates to the -1 to 1 range.
	float convertedX = 2.0f * (((float)threadWidthPos - ((float)currentWidth * 0.5f)) / (float)currentWidth);
	float convertedY = ((float)currentHeight - (float)threadHeightPos * 2.0f) / (float)currentHeight;
	Vector4 pixelVSpace = Vector4(convertedX * pElement1, convertedY * pElement2, 1.0f, 0.0f);
	Vector4 tempDirectionWSpace = inverseVMatrix.Mult(pixelVSpace);
	Vector3 directionWSpace = Vector3(tempDirectionWSpace.x, tempDirectionWSpace.y, tempDirectionWSpace.z);
	directionWSpace = directionWSpace.Normalize();

	char data = ' ';
	float closest = std::numeric_limits<float>::max();
	Object* closestObject = nullptr;
	for (size_t i = 0; i < objectNr; i++)
	{
		//Ray-Sphere intersection test.
		if (culledObjects[i]->GetTag() == "Sphere") //Make a switch instead
		{
			Vector3 objectToCam = cameraPos - culledObjects[i]->GetPos();
			float radius = reinterpret_cast<Sphere*>(culledObjects[i])->GetRadius();

			float a = Dot(directionWSpace, directionWSpace);
			float b = 2.0f * Dot(directionWSpace, objectToCam);
			float c = Dot(objectToCam, objectToCam) - (radius * radius);

			float discriminant = b*b - 4.0f*a*c;

			//It hit
			if (discriminant >= 0.0f)
			{
				float t1 = (-b + sqrt(discriminant)) / (2.0f * a);
				float t2 = (-b - sqrt(discriminant)) / (2.0f * a);

				float closerPoint = 0.0f;
				if (t1 <= t2)
				{
					closerPoint = t1;
				}
				else
				{
					closerPoint = t2;
				}

				if (closerPoint < closest)
				{
					closest = closerPoint;
					closestObject = culledObjects[i];
				}
			}
		}
	}

	//If it didnt hit anything.
	if (!closestObject)
	{
		//PrintMachine::GetInstance()->SendData(threadWidthPos, threadHeightPos, data);
		(*twoDArray)[threadHeightPos][threadWidthPos] = data;
		return;
	}
	

	if (closestObject->GetTag() == "Sphere")
	{
		//Calculate light etc and then set data.
		data = '#';
	}

	(*twoDArray)[threadHeightPos][threadWidthPos] = data;
	//PrintMachine::GetInstance()->SendData(threadWidthPos, threadHeightPos, data);
	return;
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
	PrintMachine::GetInstance()->Fill('s'); //Temp
	m_camera->Init();
	m_scene->Init();
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
		PrintMachine::GetInstance()->Print(m_scene->SendCulledObjects()[0]->GetPos().y);
		//m_frameTimer = 0.0f;
	//}
	
	return true;
}

void Engine::Render()
{
	m_camera->Update();
	m_scene->Update();

	std::vector<Object*> culledObjects = m_scene->SendCulledObjects();
	size_t objectNr = culledObjects.size();
	//Like our pixel shader.
	std::vector<std::thread> pixels;
	Matrix inverseVMatrix = m_camera->GetVMatrix(); //m_camera->GetInverseVMatrix();
	size_t x = PrintMachine::GetInstance()->GetWidth();
	size_t y = PrintMachine::GetInstance()->GetHeight();
	float element1 = m_camera->GetPMatrix().row1.x;
	float element2 = m_camera->GetPMatrix().row2.y;
	Vector3 camPos = m_camera->GetPos();
	std::vector<std::vector<char>>* TwoDArray = PrintMachine::GetInstance()->Get2DArray();
	std::cout << sizeof(Object) << " " << sizeof(Sphere);
	for (size_t i = 0; i < y; i++)
	{
		for (size_t j = 0; j < x; j++)
		{
			
			pixels.push_back(std::thread(
				calculatePixel,
				//Arguments to pixel shader.
				inverseVMatrix, //Need the inverse here
				element1,
				element2,
				camPos,
				culledObjects,
				objectNr,
				TwoDArray,
				x,
				y,
				j,
				i
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

