#include "pch.h"
#include "Sphere.h"

__host__ __device__ float Sphere::GetRadius()
{
	return m_radius;
}

__device__ void Sphere::Update(long double deltaTime)
{
	m_middlePos.y += speed * mover * deltaTime;
	if (m_middlePos.y < -10.0f || m_middlePos.y > 10.0f)
	{
		mover *= -1;
	}
}