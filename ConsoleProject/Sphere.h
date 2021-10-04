#pragma once
#include "Object.h"

class Sphere : public Object
{
public:
	Sphere(Vector3 middle, float r) :
		Object{ middle, "Sphere" }, m_radius{ r }, mover{ -1 }
	{
	}
	virtual ~Sphere() noexcept = default;
	void Update();
	float GetRadius();
private:
	float m_radius;

	int mover;
};