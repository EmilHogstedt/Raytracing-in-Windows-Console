#include "pch.h"
#include "Plane.h"

Plane::Plane(const MyMath::Vector3& center, const MyMath::Vector3& normal, const MyMath::Vector3& color) : Object3D(center, ObjectType::PlaneType, color)
{
	m_normal = normal;
}

__device__
void Plane::Update(long double dt)
{
	return;
}

__host__ __device__
const MyMath::Vector3& Plane::GetNormal()
{
	return m_normal;
}