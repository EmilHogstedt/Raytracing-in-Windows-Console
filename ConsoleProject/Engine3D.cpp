#include "pch.h"
#include "Engine3D.h"

//A lot of this will be moved/removed when CUDA is added.
//Move input stuff to an input handler.
Engine3D* Engine3D::pInstance{ nullptr };
Time* Engine3D::m_timer{ nullptr };
Camera3D* Engine3D::m_camera{ nullptr };
Scene3D* Engine3D::m_scene{ nullptr };
long double Engine3D::m_frameTimer = 0.0f;
long double Engine3D::m_fpsTimer = 0.0f;
int Engine3D::m_fps = 0;

size_t Engine3D::m_num_threads = 0;
bool Engine3D::m_lockMouse = false;
bool Engine3D::m_mouseJustMoved = false;
RayTracingParameters* Engine3D::m_deviceRayTracingParameters = nullptr;

bool Engine3D::m_currentRenderingBuffer = true;
std::thread Engine3D::m_gpuThread;
std::deque<Engine3D::JobHolder*> Engine3D::m_renderingQueue;
std::mutex Engine3D::m_queueMutex;
std::condition_variable Engine3D::m_renderingCondition;
bool Engine3D::m_terminateRendering = false;
bool Engine3D::m_stopped = false;

std::mutex Engine3D::m_swapchainMutex1;
std::mutex Engine3D::m_swapchainMutex2;

void Engine3D::WaitForRenderingJob()
{
	Engine3D* myself = GetInstance();
	JobHolder* Job;
	while (!m_terminateRendering)
	{
		{
			std::unique_lock<std::mutex> lock(m_queueMutex);

			m_renderingCondition.wait(lock, [&, myself]()
				{
					return !m_renderingQueue.empty() || m_terminateRendering;
				});
			if (m_terminateRendering)
			{
				break;
			}

			Job = m_renderingQueue.front();
			m_renderingQueue.pop_front();
		}
		Job->m_Job(Job->m_x, Job->m_y, Job->m_element1, Job->m_element2, Job->m_deviceObjects, Job->m_deviceParams, Job->m_deviceResultArray, Job->m_dt, Job->m_hostResultArray);
		delete Job;
	}
}

//Used by the main thread to add jobs into different queues for the worker threads to take from.

void Engine3D::AddJob(
	std::function<void(size_t x, size_t y, float element1, float element2, DeviceObjectArray<Object3D*> deviceObjects, RayTracingParameters* deviceParams, char* deviceResultArray, double dt, char* hostResultArray)> newJob,
	size_t x, size_t y, float element1, float element2, DeviceObjectArray<Object3D*> deviceObjects, RayTracingParameters* deviceParams, char* deviceResultArray, double dt, char* hostResultArray
	)
{
	{
		std::unique_lock<std::mutex> lock(m_queueMutex);
		size_t currentSize = m_renderingQueue.size();
		m_renderingQueue.resize(currentSize + x);
		m_renderingQueue[currentSize] = DBG_NEW JobHolder(newJob, x, y, element1, element2, deviceObjects, deviceParams, deviceResultArray, dt, hostResultArray);
	}
	m_renderingCondition.notify_one();
}

//Called at the end of the program to make sure the cpu threads get shut down correctly.
void Engine3D::ShutdownRenderingThread()
{
	{
		std::unique_lock<std::mutex> lock(m_queueMutex);
		m_terminateRendering = true;
	}

	m_renderingCondition.notify_all();

	m_gpuThread.join();

	m_stopped = true;
}

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
	if (!m_stopped)
		ShutdownRenderingThread();
	delete m_timer;
	delete m_camera;
	delete m_scene;
	cudaFree(m_deviceRayTracingParameters);
}

void Engine3D::Start()
{
	PrintMachine::CreatePrintMachine(250, 100, &m_swapchainMutex1, &m_swapchainMutex2);
	m_camera->Init();
	m_camera->Update();
	m_scene->Init();

	//Initialize threads
	m_num_threads = std::thread::hardware_concurrency();
	m_gpuThread = std::thread(WaitForRenderingJob);
}

bool Engine3D::Run()
{
	//Check if the console window is still running.
	if (!PrintMachine::GetInstance()->CheckIfRunning())
	{
		CleanUp();
		return false;
	}
	
	long double dt = m_timer->DeltaTime();
	CheckKeyboard(dt);
	//Move using the current input.
	m_camera->Move(dt);

	m_timer->Update();
	m_fps++;

	m_frameTimer += dt;
	m_fpsTimer += m_timer->DeltaTime();

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
	PrintMachine::GetInstance()->Print();

	//Some debugging text. Maybe add an information panel at the bottom that can get sent text? Should be done in PrintMachine though.
	Vector3 pos = m_camera->GetPos();
	Vector3 rot = m_camera->GetRot();
	std::cout << "Pos: " << pos.x << " " << pos.y << " " << pos.z << std::endl;
	std::cout << "Rot: " << rot.x << " " << rot.y << " " << rot.z << std::endl;
	COORD coords = m_camera->GetMouseCoords();
	std::cout << "Mousecoords: " << coords.X << " " << coords.Y << std::endl;
	
	return true;
}

void Engine3D::Render()
{
	//Update the view matrix.
	m_camera->Update();
	//Update the objects in the scene. Not needed as it is updated on GPU now.
	//m_scene->Update(m_timer->DeltaTime());

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
	
	DeviceObjectArray<Object3D*> objects = m_scene->GetObjects();
	AddJob(&RayTracingWrapper, x, y, element1, element2, objects, m_deviceRayTracingParameters, PrintMachine::GetInstance()->GetDeviceBuffer(), m_timer->DeltaTime(), PrintMachine::GetInstance()->GetCurrentHostBuffer(m_currentRenderingBuffer));
	m_currentRenderingBuffer = !m_currentRenderingBuffer;
	//RayTracingWrapper(x, y, element1, element2, objects, m_deviceRayTracingParameters, PrintMachine::GetInstance()->GetDeviceBuffer(), m_timer->DeltaTime());
}

//Move this to an input handler.
void Engine3D::CheckKeyboard(long double dt)
{
	INPUT_RECORD event;
	HANDLE consoleHandle = PrintMachine::GetInstance()->GetConsoleHandle();
	DWORD count = 0;

	//Rewrite key presses. https://stackoverflow.com/questions/41600981/how-do-i-check-if-a-key-is-pressed-on-c
	GetNumberOfConsoleInputEvents(consoleHandle, &count);
	if (count > 0)
	{
		ReadConsoleInput(PrintMachine::GetInstance()->GetConsoleHandle(), &event, 1, &count);

		if (event.EventType == KEY_EVENT)
		{
			switch (event.Event.KeyEvent.wVirtualKeyCode)
			{
			//W key
			case 0x57:
			{
				if (event.Event.KeyEvent.bKeyDown)
				{
					m_camera->m_Keys.W = 1;
				}
				else
				{
					m_camera->m_Keys.W = 0;
				}
				break;
			}
			//A key
			case 0x41:
			{
				if (event.Event.KeyEvent.bKeyDown)
				{
					m_camera->m_Keys.A = 1;
				}
				else
				{
					m_camera->m_Keys.A = 0;
				}
				break;
			}
			//S key
			case 0x53:
			{
				if (event.Event.KeyEvent.bKeyDown)
				{
					m_camera->m_Keys.S = 1;
				}
				else
				{
					m_camera->m_Keys.S = 0;
				}
				break;
			}
			//D key
			case 0x44:
			{
				if (event.Event.KeyEvent.bKeyDown)
				{
					m_camera->m_Keys.D = 1;
				}
				else
				{
					m_camera->m_Keys.D = 0;
				}
				break;
			}
			//Shift key
			case VK_SHIFT:
			{
				if (event.Event.KeyEvent.bKeyDown)
				{
					m_camera->m_Keys.Shift = 1;
				}
				else
				{
					m_camera->m_Keys.Shift = 0;
				}
				break;
			}
			//Spacebar key
			case VK_SPACE:
			{
				if (event.Event.KeyEvent.bKeyDown)
				{
					m_camera->m_Keys.Space = 1;
				}
				else
				{
					m_camera->m_Keys.Space = 0;
				}
				break;
			}
			//Escape key
			case VK_ESCAPE:
			{
				if (m_lockMouse)
				{
					m_lockMouse = false;
				}
				break;
			}
			default:
			{
				break;
			}
			}

		}
		else if (event.EventType == MOUSE_EVENT)
		{
			//Mouse moving.
			if (event.Event.MouseEvent.dwEventFlags == MOUSE_MOVED)
			{
				//COORD newCoords = event.Event.MouseEvent.dwMousePosition;
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
					SetCursorPos(1000, 500); //Just do this in SetMouseCoords lol.
					newCoords.X = 1000;
					newCoords.Y = 500;
				}
				
				m_camera->AddRot(diffY, diffX, 0, dt);
				m_camera->SetMouseCoords(newCoords);
			}

			//Mouse buttons.
			if (event.Event.MouseEvent.dwButtonState == FROM_LEFT_1ST_BUTTON_PRESSED)
			{
				if (!m_lockMouse)
				{
					m_lockMouse = true;
					return; //Maybe move this.
				}
				//Do other stuff when left-clicking.
			}
		}
	}
}

void Engine3D::CleanUp()
{
	delete pInstance;
	PrintMachine::GetInstance()->CleanUp();
}