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

__device__ bool RayAABBIntersect(Vector3 bMin, Vector3 bMax, Vector3 rayOrigin, Vector3 rayDir)
{
	float t1 = (bMin.x - rayOrigin.x) / rayDir.x;
	float t2 = (bMax.x - rayOrigin.x) / rayDir.x;
	float t3 = (bMin.y - rayOrigin.y) / rayDir.y;
	float t4 = (bMax.y - rayOrigin.y) / rayDir.y;
	float t5 = (bMin.z - rayOrigin.z) / rayDir.z;
	float t6 = (bMax.z - rayOrigin.z) / rayDir.z;

	float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
	float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

	// if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behing us
	if (tmax < 0) {
		return false;
	}

	// if tmin > tmax, ray doesn't intersect AABB
	if (tmin > tmax) {
		return false;
	}

	return true;
}

__device__ float RaySphereIntersect(Vector3 spherePos, Vector3 objectToCam, float radius, Vector3 directionWSpace, float fourA, float divTwoA, float closest)
{
	float b = 2.0f * Dot(directionWSpace, objectToCam);
	float c = Dot(objectToCam, objectToCam) - (radius * radius);

	float discriminant = b * b - fourA * c;

	//It hit
	if (discriminant >= 0.0f)
	{
		float sqrtDiscriminant = sqrt(discriminant);
		float minusB = -b;
		float t1 = (minusB + sqrtDiscriminant) * divTwoA;
		float t2 = (minusB - sqrtDiscriminant) * divTwoA;

		//Remove second condition to enable "backface" culling for spheres. IE; not hit when inside them.
		if (t1 > t2 && t2 >= 0.0f)
		{
			t1 = t2;
		}

		if (t1 < closest && t1 > 0.0f)
		{
			return t1;
		}
	}

	return -1.0f;
}