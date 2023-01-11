#pragma once
#include "Sphere.h"
#include "Plane.h"
#include "GridCell.h"
#include "PointLight.h"

class Scene3D
{
public:
	Scene3D() = default;
	~Scene3D() = default;

	void Init();
	//Update all objects
	void Update(long double);

	void CleanUp();

	void CreatePlane(Vector3 middlePos, Vector3 normal, Vector3 color);
	void CreateSphere(float radius, Vector3 middlePos, Vector3 color);
	void CreatePointLight(float radius, Vector3 pos, Vector3 color);

	DeviceObjectArray<Object3D*> GetObjects();
	DeviceObjectArray<PointLight> GetPointLights();

	GridCell* GetGrid();
private:
	void CreateGrid(unsigned int size);


	//Are these hostarrays even needed?
	//Sphere* m_hostSpheres;
	//Plane* m_hostPlanes;

	DeviceObjectArray<Object3D*> m_deviceObjects;
	DeviceObjectArray<Plane> m_devicePlanes;
	DeviceObjectArray<Sphere> m_deviceSpheres;
	
	DeviceObjectArray<PointLight> m_devicePointLights;

	const unsigned int m_gridSize = GRID_DIMENSIONS;
	GridCell* m_deviceGrid = { 0 };
};