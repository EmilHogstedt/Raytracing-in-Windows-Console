#pragma once
#include "Sphere.h"
#include "Plane.h"

class Scene3D
{
public:
	Scene3D() = default;
	~Scene3D() = default;

	void Init();
	//Update all objects
	void Update(long double);

	void CleanUp();

	//TODO: Implement culling.
	std::vector<Object3D*>* SendCulledObjects();
private:
	std::vector<Object3D*> m_objects;
};