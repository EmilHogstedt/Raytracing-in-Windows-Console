#pragma once
#include "MyMath.h"


class __align__(16) PointLight
{
public:
	PointLight() noexcept : m_pos{ Vector3() }, m_rangeOfInfluence{ 0.0f }, m_color{ Vector3(1.0f, 1.0f, 1.0f) } {
	}
	PointLight(Vector3 pos, float range, Vector3 color) : m_pos{ pos }, m_rangeOfInfluence{ range }, m_color{ color } {
	}
	~PointLight() noexcept = default;

	__device__ void Update(float dt);

	__device__ const Vector3& GetPos() const noexcept
	{
		return m_pos;
	}

	__device__ float GetRange() const noexcept
	{
		return m_rangeOfInfluence;
	}

private:
	Vector3 m_pos;
	Vector3 m_color;
	float m_rangeOfInfluence;
};