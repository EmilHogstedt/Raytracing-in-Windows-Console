#include "pch.h"
#include "Scene.h"

void Scene::Init()
{
	//Kolla varför det inte funkar i andra ordningen? Padding i plane?
	m_objects.push_back(DBG_NEW Plane(Vector3(0.0f, -3.0f, 0.0f), Vector3(0.0f, 1.0f, 0.0f)));
	
	m_objects.push_back(DBG_NEW Sphere(Vector3(0.0f, 10.0f, 20.0f), 7.0f));
	m_objects.push_back(DBG_NEW Sphere(Vector3(5.0f, 10.0f, 20.0f), 3.0f));
	m_objects.push_back(DBG_NEW Sphere(Vector3(-5.0f, 10.0f, 20.0f), 6.0f));
	m_objects.push_back(DBG_NEW Sphere(Vector3(5.0f, 10.0f, 40.0f), 10.0f));
	m_objects.push_back(DBG_NEW Sphere(Vector3(-5.0f, 10.0f, 40.0f), 4.0f));
}

//Update all objects
void Scene::Update(long double deltaTime)
{
	for (auto object : m_objects)
	{
		object->Update(deltaTime);
	}
}

void Scene::CleanUp()
{
	for (auto r : m_objects)
	{
		delete r;
	}
}

std::vector<Object*>* Scene::SendCulledObjects()
{
	return &m_objects;
}