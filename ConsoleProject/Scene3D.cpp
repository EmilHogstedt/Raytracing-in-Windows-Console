#include "pch.h"
#include "Scene3D.h"

//Here the starter objects are created.
void Scene3D::Init()
{
	m_objects.push_back(DBG_NEW Sphere(Vector3(0.0f, 10.0f, 20.0f), 7.0f));
	m_objects.push_back(DBG_NEW Sphere(Vector3(5.0f, 10.0f, 20.0f), 3.0f));
	m_objects.push_back(DBG_NEW Sphere(Vector3(-5.0f, 10.0f, 20.0f), 6.0f));
	m_objects.push_back(DBG_NEW Sphere(Vector3(5.0f, 10.0f, 40.0f), 10.0f));
	m_objects.push_back(DBG_NEW Sphere(Vector3(-5.0f, 10.0f, 40.0f), 4.0f));
	m_objects.push_back(DBG_NEW Plane(Vector3(0.0f, -3.0f, 0.0f), Vector3(0.0f, 1.0f, 0.0f)));
}

//Update all objects
void Scene3D::Update(long double deltaTime)
{
	for (auto object : m_objects)
	{
		object->Update(deltaTime);
	}
}

void Scene3D::CleanUp()
{
	for (auto r : m_objects)
	{
		delete r;
	}
}

//Implement proper culling.
std::vector<Object3D*>* Scene3D::SendCulledObjects()
{
	return &m_objects;
}