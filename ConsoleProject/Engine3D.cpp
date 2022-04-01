#include "pch.h"
#include "Engine3D.h"

//A lot of this will be moved/removed when CUDA is added.
//Move input stuff to an input handler.
Engine3D* Engine3D::pInstance{ nullptr };
Time* Engine3D::m_timer{ nullptr };
Camera3D* Engine3D::m_camera{ nullptr };
Scene3D* Engine3D::m_scene{ nullptr };
long double Engine3D::m_fpsTimer = 0.0f;
int Engine3D::m_fps = 0;
/*
std::vector<std::thread> Engine3D::m_workers;
std::deque<std::mutex> Engine3D::queueMutex;
std::deque<std::condition_variable> Engine3D::condition;
bool Engine3D::terminatePool = false;
bool Engine3D::stopped = false;
std::vector<std::deque<Engine3D::JobHolder*>> Engine3D::queues;
*/
size_t Engine3D::m_num_threads = 0;
bool Engine3D::m_lockMouse = false;
bool Engine3D::m_mouseJustMoved = false;
RayTracingParameters* Engine3D::m_deviceRayTracingParameters = nullptr;
/*
//Used as a waiting area for the CPU threads until they get something to do. They then do that job.
void Engine3D::WaitForJob(
	float pElement1,
	float pElement2,
	std::vector<Object3D*>* culledObjects,
	size_t objectNr,
	size_t currentWidth,
	size_t currentHeight,
	size_t threadId
)
{
	Engine3D* myself = GetInstance();
	JobHolder* Job;
	while (!terminatePool)
	{
		{
			std::unique_lock<std::mutex> lock(queueMutex[threadId]);

			condition[threadId].wait(lock, [&, myself]()
				{
					return !queues[threadId].empty() || terminatePool;
				});
			if (terminatePool)
			{
				break;
			}

			Job = queues[threadId].front();
			queues[threadId].pop_front();
		}
		Job->m_Job(Job->m_inverseVMatrix, pElement1, pElement2, Job->m_camPos, culledObjects, objectNr, currentWidth, currentHeight, Job->m_y, Job->m_x);
		delete Job;
	}
}

//Used by the main thread to add jobs into different queues for the worker threads to take from.
void Engine3D::AddJob(std::function<void(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object3D*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadHeightPos, size_t threadWidthPos)> newJob, size_t x, size_t y, size_t threadId, Matrix inverseVMatrix, Vector3 camPos)
{
	{
		std::unique_lock<std::mutex> lock(queueMutex[threadId]);
		size_t current = queues[threadId].size();
		queues[threadId].resize(current + x);
		for (int i = 0; i < x; i++)
		{
			queues[threadId][current + i] = DBG_NEW JobHolder(newJob, i, y, inverseVMatrix, camPos);
		}
	}
	condition[threadId].notify_one();
}

//Called at the end of the program to make sure the cpu threads get shut down correctly.
void Engine3D::shutdownThreads()
{
	{
		for (size_t i = 0; i < queueMutex.size(); i++)
			std::unique_lock<std::mutex> lock(queueMutex[i]);
		terminatePool = true;
	}

	for (size_t i = 0; i < condition.size(); i++)
		condition[i].notify_all();

	for (std::thread& th : m_workers)
	{
		th.join();
	}

	m_workers.clear();

	stopped = true;
}

*/
//Creates the singleton instance.
void Engine3D::CreateEngine()
{
	if (!pInstance)
		pInstance = DBG_NEW Engine3D();
}

Engine3D* Engine3D::GetInstance()
{
	return pInstance;
}

Engine3D::Engine3D() {
	//Engine objects
	m_timer = DBG_NEW Time();
	m_camera = DBG_NEW Camera3D();
	m_scene = DBG_NEW Scene3D();
	cudaMalloc(&m_deviceRayTracingParameters, sizeof(RayTracingParameters));
}

Engine3D::~Engine3D()
{
	/*
	if (!stopped)
		shutdownThreads();
		*/
	delete m_timer;
	delete m_camera;
	delete m_scene;
	cudaFree(m_deviceRayTracingParameters);
}

void Engine3D::Start()
{
	//When we create the print machine it also starts printing.
	PrintMachine::CreatePrintMachine(250, 100);
	m_camera->Init();
	m_camera->Update();
	m_scene->Init();

	//Initialize the threads.
	size_t x = PrintMachine::GetInstance()->GetWidth();
	size_t y = PrintMachine::GetInstance()->GetHeight();
	
	/*
	std::vector<Object3D*>* culledObjects = m_scene->GetObjects();
	size_t objectNr = (*culledObjects).size();
	*/
	m_num_threads = std::thread::hardware_concurrency();
	/*
	m_workers.reserve(m_num_threads);
	//Reserve the mutex'
	queueMutex.resize(m_num_threads);
	condition.resize(m_num_threads);

	//Start the queues
	queues.reserve(m_num_threads);
	for (size_t i = 0; i < m_num_threads; i++)
	{
		queues.push_back(std::deque<Engine3D::JobHolder*>());
	}
	//Start the cpu threads.
	for (size_t i = 0; i < m_num_threads; i++)
	{
		m_workers.push_back(std::thread(
			WaitForJob,
			element1,
			element2,
			culledObjects,
			objectNr,
			x,
			y,
			i
		));
	}
	*/
}

bool Engine3D::Run()
{
	//Check if the console window is still running.
	if (!PrintMachine::GetInstance()->CheckIfRunning())
	{
		CleanUp();
		return false;
	}
	
	long double dt = m_timer->DeltaTimeRendering();
	CheckKeyboard(dt);
	//Move using the current input.
	m_camera->Move(dt);

	m_timer->UpdateRendering();
	m_fps++;

	m_fpsTimer += m_timer->DeltaTimeRendering();

	Render();

	//Once every second we update the fps.
	if (m_fpsTimer >= 1.0f)
	{
		m_scene->CreateSphere(rand() % 10, Vector3(rand() % 100 - 50, rand() % 100 - 50, rand() % 100 - 50));
		
		PrintMachine::GetInstance()->UpdateFPS(m_fps);
		m_fpsTimer = 0.0f;
		m_fps = 0;
	}
	
	//Here is the actual "painting"
	//PrintMachine::GetInstance()->Print();

	//Some debugging text. Maybe add an information panel at the bottom that can get sent text? Should be done in PrintMachine though.
	/*
	Vector3 pos = m_camera->GetPos();
	Vector3 rot = m_camera->GetRot();
	std::cout << "Pos: " << pos.x << " " << pos.y << " " << pos.z << std::endl;
	std::cout << "Rot: " << rot.x << " " << rot.y << " " << rot.z << std::endl;
	COORD coords = m_camera->GetMouseCoords();
	std::cout << "Mousecoords: " << coords.X << " " << coords.Y << std::endl;
	*/
	return true;
}

void Engine3D::Render()
{
	//Update the view matrix.
	m_camera->Update();
	//Update the objects in the scene.
	m_scene->Update(m_timer->DeltaTimeRendering());

	//Update pixel shader variables.
	//After update is complete we set the global variable
	RayTracingParameters params;
	params.inverseVMatrix = m_camera->GetInverseVMatrix();
	params.camPos = m_camera->GetPos();
	cudaMemset(m_deviceRayTracingParameters, 0, sizeof(RayTracingParameters));
	cudaMemcpy(m_deviceRayTracingParameters, &params, sizeof(RayTracingParameters), cudaMemcpyHostToDevice);

	size_t x = PrintMachine::GetInstance()->GetWidth();
	size_t y = PrintMachine::GetInstance()->GetHeight();
	float element1 = m_camera->GetPMatrix().row1.x;
	float element2 = m_camera->GetPMatrix().row2.y;
	/*
	for (size_t i = 0; i < y; i++)
	{
		size_t threadId = (size_t)std::floor((m_num_threads * i) / y);
		//Send x and add all jobs on that line.
		AddJob(&CalculatePixel, x, i, threadId, inverseVMatrix, camPos);
	}
	*/
	DeviceObjectArray<Object3D*> objects = m_scene->GetObjects();
	RayTracingWrapper(x, y, element1, element2, objects, m_deviceRayTracingParameters, PrintMachine::GetInstance()->GetDeviceBackBuffer(), PrintMachine::GetInstance()->GetBackBufferMutex(), m_timer->DeltaTimeRendering());
}

//Move this to an input handler.
void Engine3D::CheckKeyboard(long double dt)
{
	INPUT_RECORD event;
	HANDLE consoleHandle = PrintMachine::GetInstance()->GetConsoleHandle();
	DWORD count = 0;

	//Keyboard input.
	if (GetKeyState('W') & 0x8000)
	{
		m_camera->m_Keys.W = 1;
	}
	else
	{
		m_camera->m_Keys.W = 0;
	}

	if (GetKeyState('A') & 0x8000)
	{
		m_camera->m_Keys.A = 1;
	}
	else
	{
		m_camera->m_Keys.A = 0;
	}

	if (GetKeyState('S') & 0x8000)
	{
		m_camera->m_Keys.S = 1;
	}
	else
	{
		m_camera->m_Keys.S = 0;
	}

	if (GetKeyState('D') & 0x8000)
	{
		m_camera->m_Keys.D = 1;
	}
	else
	{
		m_camera->m_Keys.D = 0;
	}

	if (GetKeyState('P') & 0x8000)
	{
		if (!m_lockMouse)
		{
			m_lockMouse = true;
		}
	}

	if (GetKeyState(VK_SHIFT) & 0x8000)
	{
		m_camera->m_Keys.Shift = 1;
	}
	else
	{
		m_camera->m_Keys.Shift = 0;
	}

	if (GetKeyState(VK_SPACE) & 0x8000)
	{
		m_camera->m_Keys.Space = 1;
	}
	else
	{
		m_camera->m_Keys.Space = 0;
	}

	if (GetKeyState(VK_ESCAPE) & 0x8000)
	{
		
		if (m_lockMouse)
		{
			m_lockMouse = false;
		}
	}

	//Mouse input.
	if (GetKeyState(VK_LBUTTON) & 0x8000)
	{
		/* Dont lock the mouse with left mouse button anymore.
		if (!m_lockMouse)
		{
			m_lockMouse = true;
		}*/
		//Do other stuff when left-clicking.
	}

	//Mouse positioning.
	POINT newCoordsPoint;
	GetCursorPos(&newCoordsPoint);
	COORD newCoords;
	newCoords.X = newCoordsPoint.x;
	newCoords.Y = newCoordsPoint.y;
	if (newCoords.X == 1000 && newCoords.Y == 500)
	{
		return;
	}
	COORD oldCoords = m_camera->GetMouseCoords();
	if (oldCoords.X < 0.0f && oldCoords.Y < 0.0f)
	{
		oldCoords = newCoords;
	}

	short diffX = oldCoords.X - newCoords.X;
	short diffY = oldCoords.Y - newCoords.Y;

	if (m_lockMouse)
	{
		if (newCoords.X > 1300 || newCoords.X < 1100 || newCoords.Y > 600 || newCoords.Y < 400)
		{
			SetCursorPos(1200, 500);
			newCoords.X = 1200;
			newCoords.Y = 500;
		}
	}

	m_camera->AddRot(diffY, diffX, 0);
	m_camera->SetMouseCoords(newCoords);
}

void Engine3D::CleanUp()
{
	delete pInstance;
	PrintMachine::GetInstance()->CleanUp();
}