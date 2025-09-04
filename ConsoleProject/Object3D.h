#pragma once

#include "MyMath.h"

//Used for sending objects to the GPU.
template<typename T>
struct alignas(32) DeviceObjectArray {
	T DEVICE_MEMORY_PTR m_deviceArray;

	unsigned int allocatedBytes;
	unsigned int count;
};

enum class ObjectType { None = 0, PlaneType, SphereType };

struct alignas(32) ObjectTraceInputData
{
	MyMath::Vector3 origin;
	MyMath::Vector3 direction;

	//Only stored here to avoid recalculating these values for every object.
	float a = 0.0f;
	float fourA = 0.0f;
	float divTwoA = 0.0f;
};

struct alignas(32) ObjectTraceReturnData
{
	bool bHit = false;
	MyMath::Vector3 normal;
	float distance = 99999999.f;
};

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

	//Cuda does not support virtual functions.
	//__device__
	//virtual void Trace(const ObjectTraceInputData& inputData, ObjectTraceReturnData& returnData) const = 0;

protected:
	MyMath::Vector3 m_center;
	ObjectType m_type;
	MyMath::Vector3 m_color;
};