#pragma once
#include "Object3D.h"

class Sphere : public Object3D
{
public:
	Sphere(Vector3 middle, float r) :
		Object3D{ middle, "Sphere" }, m_radius{ r }, mover{ -1 }
	{
	}
	virtual ~Sphere() noexcept = default;
	void Update(long double);
	float GetRadius();
private:
	float m_radius;

	int mover;

	double padding[6];
	float padding2;
};