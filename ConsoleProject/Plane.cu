#include "pch.h"
#include "Plane.h"

__device__ void Plane::Update(long double dt)
{
	return;
}

__host__ __device__ Vector3 Plane::GetNormal()
{
	return m_normal;
}