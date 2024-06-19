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

	void CreatePlane(const MyMath::Vector3& middlePos, const MyMath::Vector3& normal, const MyMath::Vector3& color);
	void CreateSphere(const float radius, const MyMath::Vector3& middlePos, const MyMath::Vector3& color);

	DeviceObjectArray<Object3D*> GetObjects();

private:
	//Pointers to all objects allocated on the device.
	DeviceObjectArray<Object3D DEVICE_MEMORY_PTR> m_deviceObjects;

	//#todo: Investigate if it is possible to optimize these away and only use m_deviceObjects.
	DeviceObjectArray<Plane> m_devicePlanes;
	DeviceObjectArray<Sphere> m_deviceSpheres;
};