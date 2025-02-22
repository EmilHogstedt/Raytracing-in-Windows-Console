#include "MyMath.h"

class Object3D;

template<typename T>
struct DeviceObjectArray;

struct RayTracingParameters
{
	MyMath::Matrix inverseVMatrix;
	MyMath::Vector3 camPos;

	size_t x;
	size_t y;
	float element1;
	float element2;
	float camFarDist;
};

class RayTracingManager
{
public:
	enum RenderingMode { ASCII = 0, PIXEL, RGB_ASCII, RGB_PIXEL, RGB_NORMALS, SDL };

	RayTracingManager();

	~RayTracingManager();

	void Update(
		const RayTracingParameters& params,
		const DeviceObjectArray<Object3D*>& objects,
		double dt
	);

	void SetRenderingMode(const RenderingMode newRenderMode);

private:
	void ResetDeviceBackBuffer();

	size_t MinimizeResults(const size_t size, const size_t x, const size_t y);
	size_t Minimize8bit(const size_t size, const size_t x, const size_t y);
	size_t MinimizeRGB(const size_t size, const size_t x, const size_t y);

	char DEVICE_MEMORY_PTR m_deviceResultArray;

	std::unique_ptr<char[]> m_hostResultArray;
	std::unique_ptr<char[]> m_minimizedResultArray;

	//The raytracingparameters is always nullptr on the CPU. Meaning "new" is never called on it.
	RayTracingParameters DEVICE_MEMORY_PTR m_deviceRayTracingParameters;

	RenderingMode currentRenderingMode = ASCII;
};