#pragma once
#include "Sphere.h"
#include "Plane.h"

#define FIVE_MEGABYTES 5'000'000
#define HUNDRED_MEGABYTES 100'000'000

class Scene3D
{
public:
	Scene3D() = default;
	~Scene3D() = default;

	void Init();

	//Update all objects
	void Update(const long double dt);

	void CleanUp();

	void CreatePlane(const MyMath::Vector3& middlePos, const MyMath::Vector3& normal, const MyMath::Vector3& color, const float width, const float height);
	void CreateSphere(const float radius, const MyMath::Vector3& middlePos, const MyMath::Vector3& color);

	DeviceObjectArray<Object3D*> GetObjects();

private:
	//Checks that enough memory is allocated for adding another object in the deviceobjects pointer arrayg. If not, it allocates more space on the GPU.
	void CheckDeviceObjectsPtrMemory();

	//Checks that enough memory is allocated for adding another object in the specific object type's array. If not, it allocates more space on the GPU.
	template<typename T>
	bool CheckDeviceObjectsDataMemory(DeviceObjectArray<T>& deviceObjects);

	//Pointers to all objects allocated on the device.
	DeviceObjectArray<Object3D DEVICE_MEMORY_PTR> m_deviceObjects;

	//#todo: Maybe implement a memory allocator/manager that uses the same block of memory for all objects on the GPU?
	//All the object data.
	DeviceObjectArray<Plane> m_devicePlanes;
	DeviceObjectArray<Sphere> m_deviceSpheres;
};