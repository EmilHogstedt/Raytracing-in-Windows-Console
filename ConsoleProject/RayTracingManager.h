#include "Sphere.h"
#include "Plane.h"
#include "PrintMachine.h"

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
	RayTracingManager();

	~RayTracingManager();

	void Update(
		const RayTracingParameters& params,
		const DeviceObjectArray<Object3D*>& objects,
		double dt
	);

	enum RenderingMode { ASCII = 0, PIXEL, RGB_ASCII, RGB_PIXEL, RGB_NORMALS, SDL };

	void SetRenderingMode(const RenderingMode newRenderMode);

private:
	void ResetDeviceBackBuffer();

	size_t MinimizeResults(const size_t size, const size_t y);
	size_t Minimize8bit(const size_t size, const size_t y);
	size_t MinimizeRGB(const size_t size, const size_t y);

	char DEVICE_MEMORY_PTR m_deviceResultArray;

	std::unique_ptr<char[]> m_hostResultArray;
	std::unique_ptr<char[]> m_minimizedResultArray;

	//The raytracingparameters is always nullptr on the CPU. Meaning "new" is never called on it.
	RayTracingParameters DEVICE_MEMORY_PTR m_deviceRayTracingParameters;

	RenderingMode currentRenderingMode = ASCII;
};

#define NUM_ASCII_CHARACTERS 68
__constant__ const char ascii[NUM_ASCII_CHARACTERS] = {
	' ', '.', '`', '^', '"',
	',', ':', ';', 'I', 'l',
	'!', 'i', '>', '<', '~',
	'+', '_', '-', '?', '*',
	']', '[', '}', '{', '1',
	')', '(', '|', '/', 't',
	'f', 'j', 'r', 'x', 'n',
	'u', 'v', 'c', 'z', 'm',
	'w', 'X', 'Y', 'U', 'J',
	'C', 'L', 'q', 'p', 'd',
	'b', 'k', 'h', 'a', 'o',
	'#', '%', 'Z', 'O', '8',
	'B', '$', '0', 'Q', 'M',
	'&', 'W', '@'
};