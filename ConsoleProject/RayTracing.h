#include "Sphere.h"
#include "Plane.h"
#include "PrintMachine.h"
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
		size_t size = sizeof(char) * PrintMachine::GetInstance()->GetMaxSize();
		m_hostResultArray = (char*)malloc(size);
		memset(m_hostResultArray, 0, size);

		m_minimizedResultArray = (char*)malloc(size);
		memset(m_minimizedResultArray, 0, size);
	};

	~RayTracer()
	{
		free(m_hostResultArray);
	};

	void RayTracingWrapper(
		size_t x, size_t y,
		float element1, float element2,
		float camFarDist,
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
};

