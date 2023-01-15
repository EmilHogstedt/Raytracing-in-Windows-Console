#pragma once
#include "Object3D.h"

class __align__(16) PointLight : public Object3D
{
public:
	PointLight(Vector3 pos, float range, Vector3 color) : Object3D{pos, ObjectType::PointLightType, color}, m_rangeOfInfluence{range} {
	}
	~PointLight() noexcept = default;

	__device__ void Update(long double);

	__device__ float GetRange() const noexcept
	{
		return m_rangeOfInfluence;
	}

private:
	float m_rangeOfInfluence;
};