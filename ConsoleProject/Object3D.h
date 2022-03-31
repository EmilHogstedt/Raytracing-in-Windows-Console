#pragma once
#include "MyMath.h"

//Used for sending objects to the GPU.
template<typename T>
struct DeviceObjectArray {
	T* m_deviceArray1;
	T* m_deviceArray2;

	bool using1st;
	unsigned int allocatedBytes;
	unsigned int count;

};

enum ObjectType { None = 0, PlaneType, SphereType };

//Barebones baseclass in order to be able to group all objects together.
class Object3D
{
public:
	Object3D() : m_middlePos{ Vector3() }, m_type { None }
	{
	}
	Object3D(Vector3 middle, ObjectType type) :
		m_middlePos{ middle }, m_type{ type }
	{
	}
	virtual ~Object3D() noexcept = default;

	//__device__ virtual void Update(long double) = 0;
	__host__ __device__ ObjectType GetType();
	__host__ __device__ Vector3 GetPos();
	void SetType(ObjectType);
	void SetMiddlePos(Vector3);
protected:
	Vector3 m_middlePos;
private:
	ObjectType m_type;
};