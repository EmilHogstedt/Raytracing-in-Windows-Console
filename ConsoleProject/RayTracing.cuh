#include "Sphere.h"
#include "Plane.h"

struct RayTracingParameters
{
	Matrix inverseVMatrix;
	Vector3 camPos;
};

void RayTracingWrapper(size_t x, size_t y, float element1, float element2, DeviceObjectArray<Object3D*> objects, RayTracingParameters* deviceParams, char* deviceResultArray, double dt);
