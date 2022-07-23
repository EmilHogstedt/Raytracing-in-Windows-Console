#include "pch.h"
#include "Intersections.h"

__device__ bool SphereAABBIntersect(Vector3 spherePos, float r2, Vector3 bMin, Vector3 bMax)
{
	/*float x = fmaxf(fminf(spherePos.x, bMax.x), bMin.x);
	float y = fmaxf(fminf(spherePos.y, bMax.y), bMin.y);
	float z = fmaxf(fminf(spherePos.z, bMax.z), bMin.z);
	Vector3 posOnCube = Vector3(x, y, z);
	Vector3 dist = posOnCube - spherePos;
	float l = sqrt(dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
	if (l < r2 / r2)
	{
		return true;
	}
	else
	{
		return false;
	}*/

	float dist = 0.0f;
	float v = spherePos.x;
	if (v < bMin.x) {
		dist += ((bMin.x - v) * (bMin.x - v));
	}
	else if (v > bMax.x) {
		dist += ((v - bMax.x) * (v - bMax.x));
	}

	v = spherePos.y;
	if (v < bMin.y) {
		dist += ((bMin.y - v) * (bMin.y - v));
	}
	else if (v > bMax.y) {
		dist += ((v - bMax.y) * (v - bMax.y));
	}

	v = spherePos.z;
	if (v < bMin.z) {
		dist += ((bMin.z - v) * (bMin.z - v));
	}
	else if (v > bMax.z) {
		dist += ((v - bMax.z) * (v - bMax.z));
	}

	return dist <= r2;
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

	// if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behind us
	if (tmax < 0) {
		return false;
	}

	// if tmin > tmax, ray doesn't intersect AABB
	if (tmin > tmax) {
		return false;
	}

	return true;
}

__device__ Vector2 RaySphereIntersect(Vector3 spherePos, Vector3 objectToCam, float radius, Vector3 directionWSpace, float fourA, float divTwoA, float closest)
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
			float temp = t1;
			t1 = t2;
			t2 = temp;
		}

		if (t1 < closest && t1 > 0.0f)
		{
			return Vector2(t1, t2);
		}
	}

	return Vector2(-1.0f, -1.0f);
}