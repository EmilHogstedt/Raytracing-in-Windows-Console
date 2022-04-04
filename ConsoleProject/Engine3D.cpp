#include "pch.h"
#include "Engine3D.h"

//A lot of this will be moved/removed when CUDA is added.
//Move input stuff to an input handler.
Engine3D* Engine3D::pInstance{ nullptr };
Camera3D* Engine3D::m_camera{ nullptr };
Scene3D* Engine3D::m_scene{ nullptr };

Time* Engine3D::m_timer{ nullptr };
long double Engine3D::m_fpsTimer = 0.0f;
int Engine3D::m_fps = 0;

size_t Engine3D::m_num_threads = 0;

bool Engine3D::m_lockMouse = false;

RayTracer* Engine3D::m_rayTracer = nullptr;
RayTracingParameters* Engine3D::m_deviceRayTracingParameters = nullptr;

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
	delete m_timer;
	delete m_camera;
	delete m_scene;
	delete m_rayTracer;
	cudaFree(m_deviceRayTracingParameters);
}

void Engine3D::Start()
{
	//When we create the print machine it also starts printing.
	PrintMachine::CreatePrintMachine(400, 100);
	m_rayTracer = DBG_NEW RayTracer();
	m_camera->Init();
	m_camera->Update();
	m_scene->Init();

	//Not needed atm. Maybe later.
	m_num_threads = std::thread::hardware_concurrency();
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
		//m_scene->CreateSphere(rand() % 10, Vector3(rand() % 100 - 50, rand() % 100 - 50, rand() % 100 - 50));
		
		PrintMachine::GetInstance()->UpdateFPS(m_fps);
		m_fpsTimer = 0.0f;
		m_fps = 0;
	}
	
	//Some debugging text. Maybe add an information panel at the bottom that can get sent text? Should be done in PrintMachine though.
	Vector3 forward = m_camera->GetForward();
	std::ostringstream oss;
	oss << "\x1b[36mForward: X: " << forward.x << " Y: " << forward.y << " Z: " << forward.z << "\x1b[m";
	PrintMachine::GetInstance()->SetDebugInfo(oss.str());
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
	//After update is complete we set the device variable
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
	m_rayTracer->RayTracingWrapper(x, y, element1, element2, objects, m_deviceRayTracingParameters, PrintMachine::GetInstance()->GetDeviceBackBuffer(), PrintMachine::GetInstance()->GetBackBufferMutex(), m_timer->DeltaTimeRendering());
}

//Move this to an input handler.
void Engine3D::CheckKeyboard(long double dt)
{
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

	//Change printing mode.
	if (GetKeyState(VK_F1) & 0x8000)
	{
		PrintMachine::GetInstance()->SetPrintMode(PrintMachine::ASCII);
	}
	if (GetKeyState(VK_F2) & 0x8000)
	{
		PrintMachine::GetInstance()->SetPrintMode(PrintMachine::Pixel);
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