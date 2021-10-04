#include "pch.h"
#include "Scene.h"

void Scene::Init()
{
	m_objects.push_back(DBG_NEW Sphere(Vector3(0.0f, 0.0f, 40.0f), 15.0f));
}

//Update all objects
void Scene::Update()
{
	for (auto object : m_objects)
	{
		object->Update();
	}
}

void Scene::CleanUp()
{
	for (auto r : m_objects)
	{
		delete r;
	}
}

std::vector<Object*> Scene::SendCulledObjects()
{
	return m_objects;
}