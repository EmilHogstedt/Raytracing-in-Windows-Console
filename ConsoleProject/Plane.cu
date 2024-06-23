#include "pch.h"
#include "Plane.h"

Plane::Plane(const MyMath::Vector3& center, const MyMath::Vector3& normal, const MyMath::Vector3& color, const float width, const float height) : Object3D(center, ObjectType::PlaneType, color)
{
	//Normalize the normal.
	m_normal = normal.Normalize();
	m_width = width;
	m_height = height;
}

__device__
void Plane::Update(long double dt)
{
	return;
}

__host__ __device__
MyMath::Vector3 Plane::GetNormal() const
{
	return m_normal;
}

__host__ __device__
float Plane::GetWidth() const
{
	return m_width;
}

__host__ __device__
float Plane::GetHeight() const
{
	return m_height;
}
