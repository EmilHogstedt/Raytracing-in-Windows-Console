#pragma once
#include "PrintMachine.h"
#include "Camera3D.h"
#include "Timer.h"
#include "Scene3D.h"

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

	static void Render();

	static void CheckKeyboard(long double dt);

	static Time* m_timer;
	static Camera3D* m_camera;
	static Scene3D* m_scene;
	static long double m_frameTimer;
	static long double m_fpsTimer;
	static int m_fps;
	static std::vector<std::thread> m_workers;
	
	struct JobHolder
	{
		JobHolder() = default;
		JobHolder(std::function<void(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object3D*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadHeightPos, size_t threadWidthPos)> newJob, size_t x, size_t y, Matrix inverseVMatrix, Vector3 camPos)
		{
			m_Job = newJob;
			m_x = x;
			m_y = y;
			m_inverseVMatrix = inverseVMatrix;
			m_camPos = camPos;
		}
		std::function<void(Matrix inverseVMatrix, float pElement1, float pElement2, Vector3 cameraPos, std::vector<Object3D*>* culledObjects, size_t objectNr, size_t currentWidth, size_t currentHeight, size_t threadHeightPos, size_t threadWidthPos)> m_Job;
		size_t m_x;
		size_t m_y;
		Matrix m_inverseVMatrix;
		Vector3 m_camPos;
	};

	//This is where the jobs are placed.
	static std::vector<std::deque<JobHolder*>> queues;
	static size_t m_num_threads;

	static bool m_lockMouse;
	static bool m_mouseJustMoved;
};
