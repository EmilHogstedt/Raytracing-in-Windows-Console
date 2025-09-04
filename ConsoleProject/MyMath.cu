#include "pch.h"
#include "MyMath.h"

__host__ __device__
float MyMath::Dot(const MyMath::Vector3& v1, const MyMath::Vector3& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__
float MyMath::Dot(const MyMath::Vector4& v1, const MyMath::Vector4& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

__host__ __device__
MyMath::Vector3 MyMath::Cross(const MyMath::Vector3& v1, const MyMath::Vector3& v2)
{
	return MyMath::Vector3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

__host__ __device__
MyMath::Vector3 MyMath::ComponentMul(const Vector3& v1, const Vector3& v2)
{
	return MyMath::Vector3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

//Generates good assembly this way.
__host__ __device__
float MyMath::Clamp(const float val, const float min, const float max)
{
	const float result = val < min ? min : val;
	return result > max ? max : result;
}

__host__ __device__
int MyMath::Clamp(const int val, const int min, const int max)
{
	const int result = val < min ? min : val;
	return result > max ? max : result;
}

__host__ __device__
bool MyMath::FloatEquals(float f1, float f2)
{
	return abs(f1 - f2) < FLT_EPSILON;
}

int MyMath::Min(int a, int b)
{
	return a < b ? a : b;
}

int MyMath::Max(int a, int b)
{
	return a < b ? b : a;
}

float MyMath::Min(float a, float b)
{
	return a < b ? a : b;
}

float MyMath::Max(float a, float b)
{
	return a < b ? b : a;
}
