#include "pch.h"
#include "Sphere.h"

__host__ __device__ float Sphere::GetRadius()
{
	return m_radius;
}

Sphere::Sphere(const MyMath::Vector3& center, const float radius, const MyMath::Vector3& color) : Object3D(center, ObjectType::SphereType, color)
{
	m_radius = radius;
	mover = -1;

	int temp = rand() % 300 + 100;
	speed = ((float)temp) / 100.0f;
}

__device__ void Sphere::Update(long double dt)
{
	m_center.y += speed * mover * dt;
	if (m_center.y < -10.0f || m_center.y > 10.0f)
	{
		m_center.y = MyMath::Clamp(m_center.y, -10.0f, 10.0f);
		mover *= -1;
	}
}