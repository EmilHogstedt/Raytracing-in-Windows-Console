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

//The "pixel shader".
void Engine3D::CalculatePixel(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object3D*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadHeightPos, size_t threadWidthPos)
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

				if (closerPoint < closest && closerPoint > 0.0f)
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

	//Dont open this. Here be dragons.
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
	//Update the objects in the scene.
	m_scene->Update(m_timer->DeltaTime());

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
	RayTracingWrapper(x, y, element1, element2, objects, m_deviceRayTracingParameters, PrintMachine::GetInstance()->GetDeviceBuffer(), m_timer->DeltaTime());
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