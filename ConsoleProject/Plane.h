#pragma once
#include "Object3D.h"

class Plane : public Object3D
{
public:
	Plane(MyMath::Vector3 point, MyMath::Vector3 normal, MyMath::Vector3 color) :
		Object3D{ point, ObjectType::PlaneType, color }, m_normal{ normal }
	{
	}
	Plane() = default;
	virtual ~Plane() noexcept = default;

	void Init(MyMath::Vector3 point, MyMath::Vector3 normal)
	{
		SetMiddlePos(point);
		SetType(ObjectType::PlaneType);
		m_normal = normal;
	}
	__device__ void Update(long double);
	__host__ __device__ MyMath::Vector3 GetNormal();
private:
	MyMath::Vector3 m_normal;
	/*
	double padding[4];
	float padding2;*/
};