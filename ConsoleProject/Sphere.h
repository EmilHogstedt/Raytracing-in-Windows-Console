#pragma once
#include "Object3D.h"

class Sphere : public Object3D
{
public:
	Sphere(const MyMath::Vector3& center, const float radius, const MyMath::Vector3& color);
	virtual ~Sphere() noexcept = default;

	__device__
	void Update(long double dt);

	__host__ __device__
	float GetRadius();

private:
	float m_radius;

	int mover;
	float speed;

	//Make sure the object has a size which is a multiple of 32.
	/*
	double padding[6];
	float padding2;
	*/
};