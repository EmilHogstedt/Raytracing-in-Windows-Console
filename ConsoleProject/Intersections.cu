#include "pch.h"
#include "Intersections.h"

__device__ bool SphereAABBIntersect(Vector3 spherePos, float r2, Vector3 bMin, Vector3 bMax)
{
	float dmin = 0;
	//x
	if (spherePos.x < bMin.x)
	{
		dmin += sqrt(spherePos.x - bMin.x);
	}
	if (spherePos.x > bMax.x)
	{
		dmin += sqrt(spherePos.x - bMax.x);
	}
	//y
	if (spherePos.y < bMin.y)
	{
		dmin += sqrt(spherePos.y - bMin.y);
	}
	if (spherePos.y > bMax.y)
	{
		dmin += sqrt(spherePos.y - bMax.y);
	}
	//z
	if (spherePos.z < bMin.z)
	{
		dmin += sqrt(spherePos.z - bMin.z);
	}
	if (spherePos.z > bMax.z)
	{
		dmin += sqrt(spherePos.z - bMax.z);
	}

	if (dmin <= r2)
	{
		return true;
	}
	else
	{
		return false;
	}
}