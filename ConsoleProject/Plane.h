#pragma once
#include "Object3D.h"

class Plane : public Object3D
{
public:
	Plane(const MyMath::Vector3& center, const MyMath::Vector3& normal, const MyMath::Vector3& color);
	Plane() = default;
	virtual ~Plane() noexcept = default;

	void Init(MyMath::Vector3 point, MyMath::Vector3 normal)
	{
		SetMiddlePos(point);
		SetType(ObjectType::PlaneType);
		m_normal = normal;
	}

	__device__
	void Update(long double);

	__host__ __device__
	const MyMath::Vector3& GetNormal();

private:
	MyMath::Vector3 m_normal;
	
	//Make sure the object has a size which is a multiple of 32.
	//int padding[2];
	//float padding2;
};