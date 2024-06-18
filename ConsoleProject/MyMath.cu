#include "pch.h"
#include "MyMath.h"

__host__ __device__
float MyMath::Dot(MyMath::Vector3 v1, MyMath::Vector3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__
float MyMath::Dot(MyMath::Vector4 v1, MyMath::Vector4 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

__host__ __device__
MyMath::Vector3 MyMath::Cross(MyMath::Vector3 v1, MyMath::Vector3 v2)
{
	return MyMath::Vector3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}