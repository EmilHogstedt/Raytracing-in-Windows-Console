#include "pch.h"
#include "Sphere.h"

#include "MyMath.h"

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

__host__ __device__ float Sphere::GetRadius() const
{
	return m_radius;
}

__device__ void Sphere::Trace(const ObjectTraceInputData& inputData, ObjectTraceReturnData& returnData) const
{
	const MyMath::Vector3 spherePos = GetPos();

	const MyMath::Vector3 objectToCam = inputData.origin - spherePos;

	const float b = 2.0f * Dot(inputData.direction, objectToCam);
	const float c = Dot(objectToCam, objectToCam) - (m_radius * m_radius);

	const float discriminant = b * b - inputData.fourA * c;

	//If it did not hit.
	if (discriminant < 0.0f)
	{
		return;
	}

	//Now we know that the ray intersects with the sphere.

	const float sqrtDiscriminant = sqrt(discriminant);
	const float minusB = -b;

	float t1 = (minusB + sqrtDiscriminant) * inputData.divTwoA;
	const float t2 = (minusB - sqrtDiscriminant) * inputData.divTwoA;

	//If either t1 or t2 is lower than zero, we are either inside the sphere or in front of it.
	//So then we count it as a miss.
	if (t1 < 0.0f || t2 < 0.0f)
	{
		return;
	}

	//Make sure t1 equals the lowest value.
	t1 = MyMath::Min(t1, t2);
	
	returnData.bHit = true;
	returnData.distance = t1;
	returnData.normal = (inputData.origin + inputData.direction * t1 - spherePos).Normalize_GPU();
}