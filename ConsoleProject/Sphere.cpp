#include "pch.h"
#include "Sphere.h"

float Sphere::GetRadius()
{
	return m_radius;
}

void Sphere::Update(long double deltaTime)
{
	m_middlePos.x += 2 * mover * deltaTime;
	if (m_middlePos.x < -10.0f || m_middlePos.x > 10.0f)
	{
		mover *= -1;
	}
}