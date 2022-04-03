#pragma once
#include "PrintMachine.h"
#include "Camera3D.h"
#include "Scene3D.h"
#include "RayTracing.h"

//3D Singleton Engine.
class Engine3D {
	static Engine3D* pInstance;
protected:
	Engine3D();
	~Engine3D();
public:
	Engine3D(Engine3D& other) = delete;
	void operator=(const Engine3D&) = delete;

	//Called at the beginning of the program to create the instance if it is not already created.
	//This is done in this function instead of in GetInstance to reduce wait time for threads.
	static void CreateEngine();
	//Returns the instance if there is one.
	static Engine3D* GetInstance();
	static void Start();
	//Is called in our game loop.
	static bool Run();
	static void CleanUp();

private:
	static void Render();

	static void CheckKeyboard(long double dt);

	static Time* m_timer;
	static Camera3D* m_camera;
	static Scene3D* m_scene;

	static long double m_fpsTimer;
	static int m_fps;
	
	static size_t m_num_threads;
	
	static bool m_lockMouse;

	static RayTracingParameters* m_deviceRayTracingParameters;
};
