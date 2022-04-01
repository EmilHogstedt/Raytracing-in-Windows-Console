#pragma once
#include "PrintMachine.h"
#include "Camera3D.h"
#include "Timer.h"
#include "Scene3D.h"
#include "RayTracing.cuh"

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
	static std::mutex m_swapchainMutex1;
	static std::mutex m_swapchainMutex2;

	static std::mutex m_queueMutex;
	static std::condition_variable m_renderingCondition;
	static bool m_terminateRendering;
	static bool m_stopped;

	static void WaitForRenderingJob();
	static void AddJob(std::function<void(size_t x, size_t y, float element1, float element2, DeviceObjectArray<Object3D*> deviceObjects, RayTracingParameters* deviceParams, char* deviceResultArray, double dt, char* hostResultArray)> newJob,
		size_t x, size_t y, float element1, float element2, DeviceObjectArray<Object3D*> deviceObjects, RayTracingParameters* deviceParams, char* deviceResultArray, double dt, char* hostResultArray);
	static void ShutdownRenderingThread();
	/*
	//Deque of mutex' used by the threads & AddJob.
	static std::deque<std::mutex> queueMutex;
	//Condition used by AddJob & Terminate to contact the threads.
	static std::deque<std::condition_variable> condition;
	//Bools to see if its time to close down.
	static bool terminatePool;
	static bool stopped;
	//The function that the threads while-loop is in. Waits for work here.
	static void WaitForJob(float pElement1, float pElement2, std::vector<Object3D*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadId);
	//Called by the program to add a pixel as a job for threads to work on.
	static void AddJob(std::function<void(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object3D*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadHeightPos, size_t threadWidthPos)>, size_t x, size_t y, size_t threadId, Matrix inverseVMatrix, Vector3 camPos);
	static void shutdownThreads();
	//Thr funciton where the actual work is done by the thread to calculate what is seen.
	static void CalculatePixel(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object3D*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadHeightPos, size_t threadWidthPos);
	*/
	static void Render();

	static void CheckKeyboard(long double dt);

	static Time* m_timer;
	static Camera3D* m_camera;
	static Scene3D* m_scene;
	static long double m_frameTimer;
	static long double m_fpsTimer;
	static int m_fps;
	
	struct JobHolder
	{
		JobHolder() = default;
		JobHolder(std::function<void(size_t x, size_t y, float element1, float element2, DeviceObjectArray<Object3D*> deviceObjects, RayTracingParameters* deviceParams, char* deviceResultArray, double dt, char* hostResultArray)> newJob,
			size_t x, size_t y, float element1, float element2, DeviceObjectArray<Object3D*> deviceObjects, RayTracingParameters* deviceParams, char* deviceResultArray, double dt, char* hostResultArray)
		{
			m_Job = newJob;
			m_x = x;
			m_y = y;
			m_element1 = element1;
			m_element2 = element2;
			m_deviceObjects = deviceObjects;
			m_deviceParams = deviceParams;
			m_deviceResultArray = deviceResultArray;
			m_dt = dt;
			m_hostResultArray = hostResultArray;
		}
		std::function<void(
			size_t x, size_t y,
			float element1, float element2,
			DeviceObjectArray<Object3D*> deviceObjects,
			RayTracingParameters* deviceParams,
			char* deviceResultArray,
			double dt,
			char* hostResultArray
		)> m_Job;
		size_t m_x;
		size_t m_y;
		float m_element1;
		float m_element2;
		DeviceObjectArray<Object3D*> m_deviceObjects;
		RayTracingParameters* m_deviceParams;
		char* m_deviceResultArray;
		double m_dt;
		char* m_hostResultArray;
	};

	//This is where the jobs are placed.
	static std::deque<JobHolder*> m_renderingQueue;
	
	static size_t m_num_threads;
	
	static bool m_currentRenderingBuffer;
	static std::thread m_gpuThread;

	static bool m_lockMouse;
	static bool m_mouseJustMoved;

	static RayTracingParameters* m_deviceRayTracingParameters;
};
