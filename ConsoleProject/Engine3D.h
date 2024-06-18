#pragma once
#include "PrintMachine.h"
#include "Camera3D.h"
#include "Scene3D.h"
#include "RayTracing.h"

class Engine3D {
public:
	Engine3D() = default;
	~Engine3D() = default;

	void Start();
	bool Run();
	void CleanUp();

private:
	void Render();

	void CheckKeyboard(long double dt);

	std::unique_ptr<Time> m_timer;
	std::unique_ptr<Camera3D> m_camera;
	std::unique_ptr<Scene3D> m_scene;

	long double m_fpsTimer;
	int m_fps;
	
	size_t m_numThreads;
	
	//Move these to input handler.
	bool m_lockMouse;

	std::unique_ptr<RayTracer> m_rayTracer;

	//The raytracingparameters is always nullptr on the CPU. Meaning "new" is never called on it.
	RayTracingParameters* m_deviceRayTracingParameters;

	bool m_quit = false;
};