#pragma once
#include "Sphere.h"
#include "Plane.h"

class Scene
{
public:
	Scene() = default;
	~Scene() = default;

	void Init();
	//Update all objects
	void Update(long double);

	void CleanUp();

	//TODO: Implement culling.
	std::vector<Object*>* SendCulledObjects();
private:
	std::vector<Object*> m_objects;
};