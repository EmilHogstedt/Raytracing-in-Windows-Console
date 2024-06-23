#include "pch.h"
#include "Engine3D.h"

void Engine3D::Start()
{
	//Engine objects
	m_timer = std::make_unique<Time>();
	m_camera = std::make_unique<Camera3D>();
	m_scene = std::make_unique<Scene3D>();

	//Allocate memory for the raytracingparameters.
	cudaMalloc(&m_deviceRayTracingParameters, sizeof(RayTracingParameters));

	//When we create the print machine it also starts printing.
	//Set it to 1920x500 for high resolution.
	//Set it to 400x150 for low resolution.
	PrintMachine::Start(400, 150);

	//The printmachine needs to be created for the raytracer to be created.
	m_rayTracer = std::make_unique<RayTracer>();

	m_camera->Init();
	m_camera->Update();

	m_scene->Init();

	//Not needed atm. Maybe later.
	m_numThreads = std::thread::hardware_concurrency();
}

bool Engine3D::Run()
{
	//Check if the console window is still running.
	if (!PrintMachine::CheckIfRunning())
	{
		CleanUp();
		return false;
	}
	
	//Check if we want to quit.
	if (m_bShouldQuit)
	{
		return false;
	}

	long double dt = m_timer->DeltaTime();
	m_timer->Update();
	m_fpsTimer += m_timer->DeltaTime();
	m_fps++;

	//Handle keyboard input
	CheckKeyboard(dt);

	//Move using the current input.
	m_camera->Move(dt);

	Render();

	//Once every second we update the fps.
	if (m_fpsTimer >= 1.0f)
	{
		//Create a sphere every second for testing purposes
		m_scene->CreateSphere(static_cast<float>(rand() % 10), MyMath::Vector3(rand() % 100 - 50, rand() % 100 - 50, rand() % 100 - 50), MyMath::Vector3(rand() % 255, rand() % 255, rand() % 255));
		
		//Send the fps to the printmachine.
		PrintMachine::UpdateRenderingFPS(m_fps);
		m_fpsTimer = 0.0f;
		m_fps = 0;
	}
	
	//Some debugging text. Maybe add an information panel at the bottom that can get sent text? Should be done in PrintMachine though.
	/*
	Vector3 forward = m_camera->GetForward();
	std::ostringstream oss;
	oss << "\x1b[36mForward: X: " << forward.x << " Y: " << forward.y << " Z: " << forward.z << "\x1b[m";
	PrintMachine::GetInstance()->SetDebugInfo(oss.str());
	*/
	return true;
}

void Engine3D::Render()
{
	//Update the view matrix.
	m_camera->Update();

	//Update the objects in the scene.
	m_scene->Update(m_timer->DeltaTime());

	//Update pixel shader variables.
	RayTracingParameters params;
	params.inverseVMatrix = m_camera->GetInverseVMatrix();
	params.camPos = m_camera->GetPos();
	params.x = PrintMachine::GetWidth();
	params.y = PrintMachine::GetHeight();
	params.element1 = m_camera->GetPMatrix().row1.x;
	params.element2 = m_camera->GetPMatrix().row2.y;
	params.camFarDist = m_camera->GetFarPlaneDistance();

	gpuErrchk(cudaMemcpy(m_deviceRayTracingParameters, &params, sizeof(RayTracingParameters), cudaMemcpyHostToDevice));
	
	DeviceObjectArray<Object3D*> objects = m_scene->GetObjects();

	//x and y have to be sent to the wrapper anyway, as they are also used on the CPU.
	m_rayTracer->RayTracingWrapper(
		params.x,
		params.y,
		objects, 
		m_deviceRayTracingParameters, 
		m_timer->DeltaTime()
	);
}

//Move this to an input handler.
void Engine3D::CheckKeyboard(const long double dt)
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
		m_bIsMouseLocked = !m_bIsMouseLocked;
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
		m_bShouldQuit = true;
	}

	//Change printing mode.
	if (GetKeyState(VK_F1) & 0x8000)
	{
		PrintMachine::SetPrintMode(PrintMachine::ASCII);
	}
	if (GetKeyState(VK_F2) & 0x8000)
	{
		PrintMachine::SetPrintMode(PrintMachine::PIXEL);
	}
	if (GetKeyState(VK_F3) & 0x8000)
	{
		PrintMachine::SetPrintMode(PrintMachine::RGB_ASCII);
	}
	if (GetKeyState(VK_F4) & 0x8000)
	{
		PrintMachine::SetPrintMode(PrintMachine::RGB_PIXEL);
	}
	if (GetKeyState(VK_F5) & 0x8000)
	{
		PrintMachine::SetPrintMode(PrintMachine::RGB_NORMALS);
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
	newCoords.X = static_cast<SHORT>(newCoordsPoint.x);
	newCoords.Y = static_cast<SHORT>(newCoordsPoint.y);
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

	if (m_bIsMouseLocked)
	{
		SetCursorPos(1200, 500);
		newCoords.X = 1200;
		newCoords.Y = 500;
	}

	m_camera->AddRot(dt, diffY, diffX, 0);
	m_camera->SetMouseCoords(newCoords);
}

void Engine3D::CleanUp()
{
	PrintMachine::CleanUp();

	cudaFree(m_deviceRayTracingParameters);
}