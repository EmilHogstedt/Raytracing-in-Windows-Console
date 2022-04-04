#pragma once
#include "Object3D.h"

class Sphere : public Object3D
{
public:
	Sphere(Vector3 middle, float r, Vector3 color) :
		Object3D{ middle, SphereType, color }, m_radius{ r }, mover{ -1 }
	{
		int temp = rand() % 300 + 100;
		speed = ((float)temp) / 100.0f;
	}
	virtual ~Sphere() noexcept = default;
	__device__ void Update(long double);
	__host__ __device__ float GetRadius();
private:
	float m_radius;

	int mover;
	float speed;
	double padding[6];
	float padding2;
};