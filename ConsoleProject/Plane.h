#pragma once
#include "Object3D.h"

class Plane : public Object3D
{
public:
	Plane(Vector3 point, Vector3 normal) :
		Object3D{ point, "Plane" }, m_normal{ normal }
	{
	}
	virtual ~Plane() noexcept = default;
	void Update(long double);
	Vector3 GetNormal();
private:
	Vector3 m_normal;

	double padding[4];
	float padding2;
};