#pragma once
#include "Sphere.h"

class Scene
{
public:
	Scene() = default;
	~Scene() = default;

	void Init();
	//Update all objects
	void Update();

	void CleanUp();

	//TODO: Implement culling.
	std::vector<Object*> SendCulledObjects();
private:
	std::vector<Object*> m_objects;
};