#pragma once
#include "Object.h"
class Scene
{
public:
	Scene() = default;
	~Scene() = default;

	//Update all objects
	void Update();
private:
	std::vector<Object*> m_objects;
};