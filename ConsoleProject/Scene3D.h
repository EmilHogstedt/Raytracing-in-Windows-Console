#pragma once
#include "Sphere.h"
#include "Plane.h"

#define FIVE_MEGABYTES 5000000
#define HUNDRED_MEGABYTES 100000000

class Scene3D
{
public:
	Scene3D() = default;
	~Scene3D() = default;

	void Init();
	//Update all objects
	void Update(long double);

	void CleanUp();

	void CreatePlane(Vector3 middlePos, Vector3 normal);
	void CreateSphere(float radius, Vector3 middlePos);

	DeviceObjectArray<Object3D*> GetObjects();
private:
	//Are these hostarrays even needed?
	Sphere* m_hostSpheres;
	Plane* m_hostPlanes;

	DeviceObjectArray<Object3D*> m_deviceObjects;
	DeviceObjectArray<Plane> m_devicePlanes;
	DeviceObjectArray<Sphere> m_deviceSpheres;
};