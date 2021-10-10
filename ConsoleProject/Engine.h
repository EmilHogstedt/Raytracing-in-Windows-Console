#pragma once
#include "pch.h"
#include "PrintMachine.h"
#include "Camera.h"
#include "Timer.h"

#include "Scene.h"

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
	static std::deque<std::mutex> queueMutex;
	static std::mutex threadpoolMutex;
	static std::deque<std::condition_variable> condition;
	static bool terminatePool;
	static bool stopped;
	static void WaitForJob(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadId);
	static void AddJob(std::function<void(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadHeightPos, size_t threadWidthPos)>, size_t x, size_t y, size_t threadId);
	static void shutdownThreads();
	static void CalculatePixel(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadHeightPos, size_t threadWidthPos);

	static void Render();

	static Time* m_timer;
	static Camera* m_camera;
	static Scene* m_scene;
	static long double m_frameTimer;
	static long double m_fpsTimer;
	static int m_fps;
	static std::vector<std::thread> m_workers;

	struct JobHolder
	{
		JobHolder() = default;
		JobHolder(std::function<void(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadHeightPos, size_t threadWidthPos)> newJob, size_t x, size_t y)
		{
			m_Job = newJob;
			m_x = x;
			m_y = y;
		}
		std::function<void(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadHeightPos, size_t threadWidthPos)> m_Job;
		size_t m_x;
		size_t m_y;
	};

	static std::vector<std::deque<JobHolder*>> queues;
	static size_t m_num_threads;
};

Engine* Engine::pInstance{ nullptr };
Time* Engine::m_timer{ nullptr };
Camera* Engine::m_camera{ nullptr };
Scene* Engine::m_scene{ nullptr };
long double Engine::m_frameTimer = 0.0f;
long double Engine::m_fpsTimer = 0.0f;
int Engine::m_fps = 0;
std::vector<std::thread> Engine::m_workers;
std::deque<std::mutex> Engine::queueMutex;
std::mutex Engine::threadpoolMutex;
std::deque<std::condition_variable> Engine::condition;
bool Engine::terminatePool = false;
bool Engine::stopped = false;
std::vector<std::deque<Engine::JobHolder*>> Engine::queues;
size_t Engine::m_num_threads = 0;
//----------------------------------------

void Engine::WaitForJob(
	Matrix inverseVMatrix,
	float pElement1,
	float pElement2,
	Vector3 cameraPos,
	std::vector<Object*>* culledObjects,
	size_t objectNr,
	size_t currentWidth,
	size_t currentHeight,
	size_t threadId
)
{
	Engine* myself = GetInstance();
	JobHolder* Job;
	while (!terminatePool)
	{
		{
			std::unique_lock<std::mutex> lock(queueMutex[threadId]);

			condition[threadId].wait(lock, [&, myself]()
				{
					return !queues[threadId].empty();
				});
			Job = queues[threadId].front();
			queues[threadId].pop_front();
		}
			Job->m_Job(inverseVMatrix, pElement1, pElement2, cameraPos, culledObjects, objectNr, currentWidth, currentHeight, Job->m_y, Job->m_x);
			delete Job;
	}
}

void Engine::AddJob(std::function<void(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadHeightPos, size_t threadWidthPos)> newJob, size_t x, size_t y, size_t threadId)
{
	//JobHolder newJobHolder;
	//newJobHolder.m_Job = newJob;
	//newJobHolder.m_x = x;
	//newJobHolder.m_y = y;

	{
		std::unique_lock<std::mutex> lock(queueMutex[threadId]);
		size_t current = queues[threadId].size();
		queues[threadId].resize(current + x);
		for (int i = 0; i < x; i++)
		{
			//newJobHolder.m_x = i;
			queues[threadId][current + i] = DBG_NEW JobHolder(newJob, i, y);
		}
	}
	condition[threadId].notify_one();
}

void Engine::shutdownThreads()
{
	{
		for(size_t i = 0; i < queueMutex.size(); i++)
			std::unique_lock<std::mutex> lock(queueMutex[i]);
		terminatePool = true;
	}

	for(size_t i = 0; i < condition.size(); i++)
		condition[i].notify_all();

	for (std::thread& th : m_workers)
	{
		th.join();
	}

	m_workers.clear();

	stopped = true;
}

void Engine::CalculatePixel(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadHeightPos, size_t threadWidthPos)
{
	float convertedY = ((float)currentHeight - (float)threadHeightPos * 2.0f) / (float)currentHeight;
	float convertedX = 2.0f * (((float)threadWidthPos - ((float)currentWidth * 0.5f)) / (float)currentWidth);

	Vector4 pixelVSpace = Vector4(convertedX * pElement1, convertedY * pElement2, 1.0f, 0.0f);
	Vector4 tempDirectionWSpace = inverseVMatrix.Mult(pixelVSpace);
	Vector3 directionWSpace = Vector3(tempDirectionWSpace.x, tempDirectionWSpace.y, tempDirectionWSpace.z);
	directionWSpace = directionWSpace.Normalize();

	char data = ' ';
	float closest = std::numeric_limits<float>::max();
	float shadingValue = 0.0f;
	for (size_t i = 0; i < objectNr; i++)
	{
		//Ray-Sphere intersection test.
		if ((*culledObjects)[i]->GetTag() == "Sphere")
		{

			Vector3 objectToCam = cameraPos - (*culledObjects)[i]->GetPos();
			float radius = reinterpret_cast<Sphere*>((*culledObjects)[i])->GetRadius();

			float a = Dot(directionWSpace, directionWSpace);
			float b = 2.0f * Dot(directionWSpace, objectToCam);
			float c = Dot(objectToCam, objectToCam) - (radius * radius);

			float discriminant = b * b - 4.0f * a * c;

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

					Vector3 normalSphere = (Vector3(cameraPos.x + directionWSpace.x * closerPoint, cameraPos.y + directionWSpace.y * closerPoint, cameraPos.z + directionWSpace.z * closerPoint) - (*culledObjects)[i]->GetPos()).Normalize();
					shadingValue = Dot(normalSphere, Vector3() - (directionWSpace.Normalize()));
				}
			}

		}
		else if ((*culledObjects)[i]->GetTag() == "Plane")
		{
			//Check if they are paralell, if not it hit.
			Vector3 planeNormal = reinterpret_cast<Plane*>((*culledObjects)[i])->GetNormal();
			float dotLineAndPlaneNormal = Dot(directionWSpace, planeNormal);
			if (dotLineAndPlaneNormal != 0.0f)
			{
				float t1 = Dot(((*culledObjects)[i]->GetPos() - cameraPos), planeNormal) / dotLineAndPlaneNormal;

				if (t1 > 0.0f)
				{
					
					if (t1 < closest)
					{
						Vector3 p = Vector3(cameraPos.x + directionWSpace.x * t1, cameraPos.y + directionWSpace.y * t1, cameraPos.z + directionWSpace.z * t1); //Overwrite * and + operator.
						if (p.x > -7.0f && p.x < 7.0f && p.z > 12.0f && p.z < 35.0f)
						{
							closest = t1;
							shadingValue = Dot(planeNormal, Vector3() - directionWSpace);
						}
					}
				}
			}
		}
	}
	
	//Dont open this.
	{
		//I warned u
	//$ @B% 8&W M#* oah kbd pqw mZO 0QL CJU YXz cvu nxr jft /| ()1 { } [ ]?- _+~ < >i!lI ; : ,"^`.
	float t = 0.01492537f;
	if (shadingValue < 0.00001f)
	{
		data = ' ';
	}
	else if (shadingValue < t * 1)
	{
		data = '.';
	}
	else if (shadingValue < t * 2)
	{
		data = '`';
	}
	else if (shadingValue < t * 3)
	{
		data = '^';
	}
	else if (shadingValue < t * 4)
	{
		data = '"';
	}
	else if (shadingValue < t * 5)
	{
		data = ',';
	}
	else if (shadingValue < t * 6)
	{
		data = ':';
	}
	else if (shadingValue < t * 7)
	{
		data = ';';
	}
	else if (shadingValue < t * 8)
	{
		data = 'I';
	}
	else if (shadingValue < t * 9)
	{
		data = 'l';
	}
	else if (shadingValue < t * 10)
	{
		data = '!';
	}
	else if (shadingValue < t * 11)
	{
		data = 'i';
	}
	else if (shadingValue < t * 12)
	{
		data = '>';
	}
	else if (shadingValue < t * 13)
	{
		data = '<';
	}
	else if (shadingValue < t * 14)
	{
		data = '~';
	}
	else if (shadingValue < t * 15)
	{
		data = '+';
	}
	else if (shadingValue < t * 16)
	{
		data = '_';
	}
	else if (shadingValue < t * 17)
	{
		data = '-';
	}
	else if (shadingValue < t * 18)
	{
		data = '?';
	}
	else if (shadingValue < t * 19)
	{
		data = '*';
	}
	else if (shadingValue < t * 20)
	{
		data = ']';
	}
	else if (shadingValue < t * 21)
	{
		data = '[';
	}
	else if (shadingValue < t * 22)
	{
		data = '}';
	}
	else if (shadingValue < t * 23)
	{
		data = '{';
	}
	else if (shadingValue < t * 24)
	{
		data = '1';
	}
	else if (shadingValue < t * 25)
	{
		data = ')';
	}
	else if (shadingValue < t * 26)
	{
		data = '(';
	}
	else if (shadingValue < t * 27)
	{
		data = '|';
	}
	else if (shadingValue < t * 28)
	{
		data = '/';
	}
	else if (shadingValue < t * 29)
	{
		data = 't';
	}
	else if (shadingValue < t * 30)
	{
		data = 'f';
	}
	else if (shadingValue < t * 31)
	{
		data = 'j';
	}
	else if (shadingValue < t * 32)
	{
		data = 'r';
	}
	else if (shadingValue < t * 33)
	{
		data = 'x';
	}
	else if (shadingValue < t * 34)
	{
		data = 'n';
	}
	else if (shadingValue < t * 35)
	{
		data = 'u';
	}
	else if (shadingValue < t * 36)
	{
		data = 'v';
	}
	else if (shadingValue < t * 37)
	{
		data = 'c';
	}
	else if (shadingValue < t * 38)
	{
		data = 'z';
	}
	else if (shadingValue < t * 39)
	{
	data = 'm';
	}
	else if (shadingValue < t * 40)
	{
		data = 'w';
	}
	else if (shadingValue < t * 41)
	{
		data = 'X';
	}
	else if (shadingValue < t * 42)
	{
		data = 'Y';
	}
	else if (shadingValue < t * 43)
	{
		data = 'U';
	}
	else if (shadingValue < t * 44)
	{
		data = 'J';
	}
	else if (shadingValue < t * 45)
	{
		data = 'C';
	}
	else if (shadingValue < t * 46)
	{
		data = 'L';
	}
	else if (shadingValue < t * 47)
	{
		data = 'q';
	}
	else if (shadingValue < t * 48)
	{
		data = 'p';
	}
	else if (shadingValue < t * 49)
	{
		data = 'd';
	}
	else if (shadingValue < t * 50)
	{
		data = 'b';
	}
	else if (shadingValue < t * 51)
	{
		data = 'k';
	}
	else if (shadingValue < t * 52)
	{
		data = 'h';
	}
	else if (shadingValue < t * 53)
	{
		data = 'a';
	}
	else if (shadingValue < t * 54)
	{
		data = 'o';
	}
	else if (shadingValue < t * 55)
	{
		data = '#';
	}
	else if (shadingValue < t * 56)
	{
		data = '%';
	}
	else if (shadingValue < t * 57)
	{
		data = 'Z';
	}
	else if (shadingValue < t * 58)
	{
		data = 'O';
	}
	else if (shadingValue < t * 59)
	{
		data = '8';
	}
	else if (shadingValue < t * 60)
	{
		data = 'B';
	}
	else if (shadingValue < t * 61)
	{
		data = '$';
	}
	else if (shadingValue < t * 62)
	{
		data = '0';
	}
	else if (shadingValue < t * 63)
	{
		data = 'Q';
	}
	else if (shadingValue < t * 64)
	{
		data = 'M';
	}
	else if (shadingValue < t * 65)
	{
		data = '&';
	}
	else if (shadingValue < t * 66)
	{
		data = 'W';
	}
	else
	{
		data = '@';
	}
	}
	PrintMachine::GetInstance()->SendData(threadWidthPos, threadHeightPos, data);
}

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
	if (!stopped)
		shutdownThreads();
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
	PrintMachine::CreatePrintMachine(150, 100);
	PrintMachine::GetInstance()->Fill('s'); //Temp
	m_camera->Init();
	m_camera->Update();
	m_scene->Init();

	
	
	//Initialize the threads.
	Matrix inverseVMatrix = m_camera->GetVMatrix(); //m_camera->GetInverseVMatrix(); Will have to be changed to a pointer.
	size_t x = PrintMachine::GetInstance()->GetWidth();
	size_t y = PrintMachine::GetInstance()->GetHeight();
	float element1 = m_camera->GetPMatrix().row1.x;
	float element2 = m_camera->GetPMatrix().row2.y;
	Vector3 camPos = m_camera->GetPos(); //Will have to be changed to a pointer
	std::vector<Object*>* culledObjects = m_scene->SendCulledObjects();
	size_t objectNr = (*culledObjects).size();
	m_num_threads = std::thread::hardware_concurrency();
	m_workers.reserve(m_num_threads);
	//Reserve the mutex'
	queueMutex.resize(m_num_threads);
	condition.resize(m_num_threads);
	
	//Start the queues
	queues.reserve(m_num_threads);
	for (size_t i = 0; i < m_num_threads; i++)
	{
		queues.push_back(std::deque<Engine::JobHolder*>());
	}

	for (size_t i = 0; i < m_num_threads; i++)
	{
		m_workers.push_back(std::thread(
			WaitForJob,
			inverseVMatrix, //Need the inverse here
			element1,
			element2,
			camPos,
			culledObjects,
			objectNr,
			x,
			y,
			i
		));
	}
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
	m_scene->Update(m_timer->DeltaTime());

	//Update pixel shader variables.
	//After update is complete we set the global variable
	size_t x = PrintMachine::GetInstance()->GetWidth();
	size_t y = PrintMachine::GetInstance()->GetHeight();
	for (size_t i = 0; i < y; i++)
	{
		size_t threadId = std::floor((m_num_threads * i) / y);
		//Send x and add all jobs on that line.
		AddJob(&CalculatePixel, x, i, threadId);
	}
	
}

void Engine::CleanUp()
{
	delete pInstance;
	PrintMachine::GetInstance()->CleanUp();
}

