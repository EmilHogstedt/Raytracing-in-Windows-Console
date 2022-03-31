#pragma once
#include "Object3D.h"

class Plane : public Object3D
{
public:
	Plane(Vector3 point, Vector3 normal) :
		Object3D{ point, PlaneType }, m_normal{ normal }
	{
	}
	Plane() = default;
	virtual ~Plane() noexcept = default;

	void Init(Vector3 point, Vector3 normal)
	{
		SetMiddlePos(point);
		SetType(PlaneType);
		m_normal = normal;
	}
	__device__ void Update(long double);
	__host__ __device__ Vector3 GetNormal();
private:
	Vector3 m_normal;

	double padding[4];
	float padding2;
};