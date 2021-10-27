#include "pch.h"
#include "Sphere.h"

float Sphere::GetRadius()
{
	return m_radius;
}

void Sphere::Update(long double deltaTime)
{
	m_middlePos.y += 2 * mover * deltaTime;
	if (m_middlePos.y < -10.0f || m_middlePos.y > 10.0f)
	{
		mover *= -1;
	}
}