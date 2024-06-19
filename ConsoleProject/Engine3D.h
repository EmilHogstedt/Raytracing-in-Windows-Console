﻿#pragma once
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

	void CheckKeyboard(const long double dt);

	std::unique_ptr<Time> m_timer;
	std::unique_ptr<Camera3D> m_camera;
	std::unique_ptr<Scene3D> m_scene;

	long double m_fpsTimer = 0.0;
	int m_fps = 0;
	
	size_t m_numThreads = 0;
	
	//Move these to input handler.
	bool m_bIsMouseLocked = false;

	std::unique_ptr<RayTracer> m_rayTracer;

	//The raytracingparameters is always nullptr on the CPU. Meaning "new" is never called on it.
	RayTracingParameters DEVICE_MEMORY_PTR m_deviceRayTracingParameters;

	bool m_bShouldQuit = false;
};