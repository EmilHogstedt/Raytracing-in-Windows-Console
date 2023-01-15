#pragma once
#include "Sphere.h"
#include "Plane.h"
#include "PrintMachine.h"
#include "GridCell.h"
#include "PointLight.h"
#include "Camera3D.h"
struct RayTracingParameters
{
	Matrix inverseVMatrix;
	Vector3 camPos;
};

class RayTracer
{
public:
	RayTracer()
	{
		cudaDeviceSetLimit(cudaLimitStackSize, GRID_DIMENSIONS * 1000);

		size_t size = sizeof(char) * PrintMachine::GetInstance()->GetMaxSize();
		m_hostResultArray = (char*)malloc(size);
		if (m_hostResultArray)
		{
			memset(m_hostResultArray, 0, size);
		}
		else
		{
			assert(false && "Could not malloc m_hostResultArray.");
		}
		
		m_minimizedResultArray = (char*)malloc(size);
		if (m_minimizedResultArray)
		{
			memset(m_minimizedResultArray, 0, size);
		}
		else
		{
			assert(false && "Could not malloc m_minimizedResultArray.");
		}

		m_lastCameraPos = Vector3(-99999, -99999, -99999);
	};

	~RayTracer()
	{
		free(m_hostResultArray);
	};

	void RayTracingWrapper(
		size_t x, size_t y,
		float element1, float element2,
		Camera3D* camera,
		GridCell* deviceGrid,
		DeviceObjectArray<Object3D*> objects,
		RayTracingParameters* deviceParams,
		char* deviceResultArray,
		std::mutex* backBufferMutex,
		double dt
	);
private:
	size_t MinimizeResults(size_t, size_t);

	char* m_hostResultArray;
	char* m_minimizedResultArray;

	Vector3 m_lastCameraPos;
};

