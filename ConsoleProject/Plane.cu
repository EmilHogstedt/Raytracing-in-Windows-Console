#include "pch.h"
#include "Plane.h"

#include "MyMath.h"

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

__device__ void Plane::Trace(const ObjectTraceInputData& inputData, ObjectTraceReturnData& returnData) const
{
	const MyMath::Vector3 planeNormal = GetNormal();
	const MyMath::Vector3 planePos = GetPos();

	const float dotLineAndPlaneNormal = Dot(inputData.direction, planeNormal);

	//Check if we are hitting the backside of the plane.
	//Check if the line and plane are paralell, if not it hit.
	if (dotLineAndPlaneNormal > 0.0f || MyMath::FloatEquals(dotLineAndPlaneNormal, 0.0f))
	{
		return;
	}

	const float t1 = Dot((planePos - inputData.origin), planeNormal) / dotLineAndPlaneNormal;

	if (t1 <= 0.0f)
	{
		return;
	}

	const MyMath::Vector3 hitPoint = inputData.origin + (inputData.direction * t1);
	const float halfPlaneWidth = GetWidth() * 0.5f;
	const float halfPlaneHeight = GetHeight() * 0.5f;

	//If the ray did not hit inbetween the width & height.
	if ((hitPoint.x <= planePos.x - halfPlaneWidth || hitPoint.x >= planePos.x + halfPlaneWidth) ||	//Width
		(hitPoint.z <= planePos.z - halfPlaneHeight || hitPoint.z >= planePos.z + halfPlaneHeight))	//Height
	{
		return;
	}

	returnData.bHit = true;
	returnData.distance = t1;
	returnData.normal = planeNormal;
}