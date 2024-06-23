#pragma once
#include "Object3D.h"

//Make sure the object has a size which is a multiple of 32.
class alignas(32) Plane : public Object3D
{
public:
	Plane(const MyMath::Vector3& center, const MyMath::Vector3& normal, const MyMath::Vector3& color, const float width, const float height);
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
	MyMath::Vector3 GetNormal() const;

	__host__ __device__
	float GetWidth() const;

	__host__ __device__
	float GetHeight() const;
private:
	MyMath::Vector3 m_normal;
	float m_width;
	float m_height;
};