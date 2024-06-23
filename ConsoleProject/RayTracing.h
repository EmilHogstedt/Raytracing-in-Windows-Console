#include "Sphere.h"
#include "Plane.h"
#include "PrintMachine.h"

struct RayTracingParameters
{
	MyMath::Matrix inverseVMatrix;
	MyMath::Vector3 camPos;

};

class RayTracer
{
public:
	RayTracer();

	~RayTracer();

	void RayTracingWrapper(
		const size_t x, const size_t y,
		const float element1, const float element2,
		const float camFarDist,
		const DeviceObjectArray<Object3D*>& objects,
		const RayTracingParameters DEVICE_MEMORY_PTR deviceParams,
		double dt
	);

private:
	void ResetDeviceBackBuffer();

	size_t MinimizeResults(const size_t size, const size_t y);
	size_t Minimize8bit(const size_t size, const size_t y);
	size_t MinimizeRGB(const size_t size, const size_t y);

	char DEVICE_MEMORY_PTR m_deviceResultArray;

	std::unique_ptr<char[]> m_hostResultArray;
	std::unique_ptr<char[]> m_minimizedResultArray;
};

