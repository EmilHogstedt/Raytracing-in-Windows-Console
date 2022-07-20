#pragma once
#include "MyMath.h"

__device__ bool SphereAABBIntersect(Vector3 spherePos, float r2, Vector3 bMin, Vector3 bMax);