#pragma once
#include "MyMath.h"

__device__ bool SphereAABBIntersect(Vector3 spherePos, float r2, Vector3 bMin, Vector3 bMax);

__device__ bool RayAABBIntersect(Vector3 min, Vector3 max, Vector3 rayOrigin, Vector3 rayDir);

__device__ Vector2 RaySphereIntersect(Vector3 spherePos, Vector3 objectTocam, float radius, Vector3 directionWSpace, float fourA, float divTwoA, float closest);