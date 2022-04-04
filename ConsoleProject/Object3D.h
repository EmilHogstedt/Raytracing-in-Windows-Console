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
	Object3D() = delete;
	Object3D(Vector3 middle, ObjectType type, Vector3 color) :
		m_middlePos{ middle }, m_type{ type }, m_color{ color }
	{
	}
	virtual ~Object3D() noexcept = default;

	//__device__ virtual void Update(long double) = 0; NOT POSSIBLE TO HAVE PURE VIRTUAL FUNCTIONS IN CUDA!!!!

	__host__ __device__ ObjectType GetType();
	__host__ __device__ Vector3 GetPos();
	__host__ __device__ Vector3 GetColor();
	void SetType(ObjectType);
	void SetMiddlePos(Vector3);
protected:
	Vector3 m_middlePos;
private:
	ObjectType m_type;
	Vector3 m_color;
};