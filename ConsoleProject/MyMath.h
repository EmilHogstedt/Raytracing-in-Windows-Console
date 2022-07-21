#pragma once

class Vector3
{
public:
	__host__ __device__ Vector3(float x, float y, float z) :
		x{ x }, y{ y }, z{ z }
	{
	}
	__host__ __device__ Vector3(int x, int y, int z) :
		x{ static_cast<float>(x) }, y{ static_cast<float>(y) }, z{ static_cast<float>(z) }
	{
	}
	__host__ __device__ Vector3()
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}
	virtual ~Vector3() noexcept = default;

	float x;
	float y;
	float z;

	__host__ __device__ Vector3(const Vector3& other)
	{
		x = other.x;
		y = other.y;
		z = other.z;
	}
	__host__ __device__ Vector3& operator=(const Vector3& other)
	{
		if (this == &other)
			return *this;

		x = other.x;
		y = other.y;
		z = other.z;

		return *this;
	}
	__host__ __device__ Vector3 operator-(const Vector3& other)
	{
		return Vector3(x - other.x, y - other.y, z - other.z);
	}
	__host__ __device__ Vector3 operator+(const Vector3& other)
	{
		return Vector3(x + other.x, y + other.y, z + other.z);
	}
	__host__ __device__ Vector3 operator+(float other)
	{
		return Vector3(x + other, y + other, z + other);
	}

	__host__ __device__ Vector3 operator/(const Vector3 & other)
	{
		return Vector3(x / other.x, y / other.y, z / other.z);
	}

	//Float operations
	__host__ __device__ Vector3 operator*(float other)
	{
		return Vector3(x * other, y * other, z * other);
	}
	__host__ __device__ void operator*=(float other)
	{
		x *= other;
		y *= other;
		z *= other;
	}
	__host__ __device__ Vector3 Normalize()
	{
		float length = 1.0f / sqrt(x * x + y * y + z * z);
		
		return Vector3(x * length, y * length, z * length);
	}

private:
};

class Vector4 : public Vector3
{
public:
	__host__ __device__ Vector4(float x, float y, float z, float w) :
		Vector3{ x, y, z }, w{ w }
	{
	}
	Vector4(Vector3 v, float w) :
		Vector3{ v }, w{ w }
	{
	}
	Vector4() :
		Vector3{ 0.0f, 0.0f, 0.0f }
	{
		w = 0.0f;
	}
	virtual ~Vector4() noexcept = default;

	float w;

	__host__ __device__ Vector4(const Vector4& other) :
		Vector3{other.x, other.y, other.z}
	{
		w = other.w;
	}

	Vector4& operator=(const Vector4& other)
	{
		if (this == &other)
			return *this;

		x = other.x;
		y = other.y;
		z = other.z;
		w = other.w;

		return *this;
	}
	
	//Implement operations (+, -, * etc)

	Vector4 Normalize()
	{
		Vector4 result = Vector4();
		float length = sqrt(x * x + y * y + z * z + w * w);
		length = 1.0f / length;
		result.x = x * length;
		result.y = y * length;
		result.z = z * length;
		result.w = w * length;

		return result;
	}

private:
};

class Matrix
{
public:
	Matrix(Vector4 v1, Vector4 v2, Vector4 v3, Vector4 v4) :
		row1{ v1 }, row2{ v2 }, row3{ v3 }, row4{ v4 }
	{
	}
	Matrix()
	{
		row1 = Vector4();
		row2 = Vector4();
		row3 = Vector4();
		row4 = Vector4();
	}
	virtual ~Matrix() noexcept = default;

	__host__ __device__ Vector4 Mult(Vector4 v)
	{
		return Vector4(
			row1.x * v.x + row1.y * v.y + row1.z * v.z + row1.w * v.w,
			row2.x * v.x + row2.y * v.y + row2.z * v.z + row2.w * v.w,
			row3.x * v.x + row3.y * v.y + row3.z * v.z + row3.w * v.w,
			row4.x * v.x + row4.y * v.y + row4.z * v.z + row4.w * v.w
			);
	}

	Vector4 row1;
	Vector4 row2;
	Vector4 row3;
	Vector4 row4;

	Matrix(const Matrix& other)
	{
		row1 = other.row1;
		row2 = other.row2;
		row3 = other.row3;
		row4 = other.row4;
	}
	Matrix& operator=(const Matrix& other)
	{
		if (this == &other)
			return *this;

		row1 = other.row1;
		row2 = other.row2;
		row3 = other.row3;
		row4 = other.row4;

		return *this;
	}

private:
};

__host__ __device__ float Dot(Vector3 v1, Vector3 v2);
__host__ __device__ float Dot(Vector4 v1, Vector4 v2);
__host__ __device__ Vector3 Cross(Vector3 v1, Vector3 v2);