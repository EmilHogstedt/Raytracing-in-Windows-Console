#pragma once

#include "MyMath.h"

//Used for sending objects to the GPU.
template<typename T>
struct DeviceObjectArray {
	T DEVICE_MEMORY_PTR m_deviceArray;

	unsigned int allocatedBytes;
	unsigned int count;
};

enum class ObjectType { None = 0, PlaneType, SphereType };


//#todo: INTRODUCE WORLDMATRIX FOR OBJECTS INSTEAD OF POSITIONS AND ROTATIONS!
//Barebones baseclass in order to be able to group all objects together.
class Object3D
{
public:
	Object3D() = delete;
	Object3D(const MyMath::Vector3& center, const ObjectType type, const MyMath::Vector3& color);
	virtual ~Object3D() noexcept = default;

	//__device__ virtual void Update(long double) = 0; NOT POSSIBLE TO HAVE PURE VIRTUAL FUNCTIONS IN CUDA!!!!

	__host__ __device__
	ObjectType GetType() const;

	__host__ __device__
	MyMath::Vector3 GetPos() const;

	__host__ __device__
	MyMath::Vector3 GetColor() const;

	void SetType(const ObjectType type);
	void SetMiddlePos(const MyMath::Vector3& center);

protected:
	MyMath::Vector3 m_center;
	ObjectType m_type;
	MyMath::Vector3 m_color;
};