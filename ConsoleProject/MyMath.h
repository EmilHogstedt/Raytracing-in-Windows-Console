#pragma once

#define FLT_EPSILON 0.000000001f

namespace MyMath
{

	class Vector3
	{
	public:
		__host__ __device__
		Vector3(const float inX, const float inY, const float inZ)
		{
			x = inX;
			y = inY;
			z = inZ;
		}

		__host__ __device__
		Vector3(const int inX, const int inY, const int inZ)
		{
			x = static_cast<float>(inX);
			y = static_cast<float>(inY);
			z = static_cast<float>(inZ);
		}

		__host__ __device__
		Vector3()
		{
			x = 0.0f;
			y = 0.0f;
			z = 0.0f;
		}

		__host__ __device__
		~Vector3() = default;

		__host__ __device__
		Vector3(const Vector3& other)
		{
			x = other.x;
			y = other.y;
			z = other.z;
		}

	public:
		//Operators
		__host__ __device__
		Vector3& operator=(const Vector3& other)
		{
			if (this == &other)
				return *this;

			x = other.x;
			y = other.y;
			z = other.z;

			return *this;
		}

		__host__ __device__
		Vector3 operator-(const Vector3& other) const
		{
			return Vector3(x - other.x, y - other.y, z - other.z);
		}

		__host__ __device__
		void operator-=(const Vector3& other)
		{
			x -= other.x;
			y -= other.y;
			z -= other.z;
		}

		__host__ __device__
		Vector3 operator+(const Vector3& other) const
		{
			return Vector3(x + other.x, y + other.y, z + other.z);
		}

		__host__ __device__
		void operator+=(const Vector3& other)
		{
			x += other.x;
			y += other.y;
			z += other.z;
		}

		//Float operations
		__host__ __device__
		Vector3 operator*(const float other) const
		{
			return Vector3(x * other, y * other, z * other);
		}

		__host__ __device__
		void operator*=(const float other)
		{
			x *= other;
			y *= other;
			z *= other;
		}

		//Copying functions
		__host__ __device__
		Vector3 Normalize() const
		{
			const float length = 1.0f / sqrt(x * x + y * y + z * z);

			return Vector3(x * length, y * length, z * length);
		}

		//In place alternatives to copying functions.
		__host__ __device__
		Vector3& Normalize_InPlace()
		{
			const float length = 1.0f / sqrt(x * x + y * y + z * z);

			x *= length;
			y *= length;
			z *= length;
			return *this;
		}

		float x;
		float y;
		float z;
	};

	class Vector4
	{
	public:
		__host__ __device__
		Vector4(const float inX, const float inY, const float inZ, const float inW)
		{
			x = inX;
			y = inY;
			z = inZ;
			w = inW;
		}

		__host__ __device__
		Vector4(const Vector3& inV, const float inW)
		{
			x = inV.x;
			y = inV.y;
			z = inV.z;
			w = inW;
		}

		__host__ __device__
		Vector4()
		{
			x = 0.0f;
			y = 0.0f;
			z = 0.0f;
			w = 0.0f;
		}

		__host__ __device__
		~Vector4() = default;

		__host__ __device__
		Vector4(const Vector4& other)
		{
			x = other.x;
			y = other.y;
			z = other.z;
			w = other.w;
		}

	public:
		//Operators
		__host__ __device__
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

		__host__ __device__
		Vector3 xyz() const
		{
			return Vector3(x, y, z);
		}

		//#todo: Implement operations (+, -, * etc)

		__host__ __device__
		Vector4 Normalize() const
		{
			const float length = 1.0f / sqrt(x * x + y * y + z * z + w * w);

			return Vector4(x * length, y * length, z * length, w * length);
		}

		//In place alternatives to copying functions.
		__host__ __device__
		Vector4& Normalize_InPlace()
		{
			const float length = 1.0f / sqrt(x * x + y * y + z * z);

			x *= length;
			y *= length;
			z *= length;
			w *= length;
			return *this;
		}

		float x;
		float y;
		float z;
		float w;
	};

	class Matrix
	{
	public:
		__host__ __device__
		Matrix(const Vector4& v1, const Vector4& v2, const Vector4& v3, const Vector4& v4)
		{
			row1 = v1;
			row2 = v2;
			row3 = v3;
			row4 = v4;
		}

		__host__ __device__
		Matrix()
		{
			row1 = Vector4();
			row2 = Vector4();
			row3 = Vector4();
			row4 = Vector4();
		}

		__host__ __device__
		~Matrix() = default;

		__host__ __device__
		Matrix(const Matrix& other)
		{
			row1 = other.row1;
			row2 = other.row2;
			row3 = other.row3;
			row4 = other.row4;
		}

		__host__ __device__
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

		__host__ __device__
		Vector4 Mult(const Vector4& v) const
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
	};

	__host__ __device__
	float Dot(const Vector3& v1, const Vector3& v2);
	__host__ __device__
	float Dot(const Vector4& v1, const Vector4& v2);
	__host__ __device__
	Vector3 Cross(const Vector3& v1, const Vector3& v2);
	//#todo: Implement cross product for Vector4.

	__host__ __device__
	float Clamp(const float val, const float min, const float max);

	__host__ __device__
	int Clamp(const int val, const int min, const int max);

	__host__ __device__
	bool FloatEquals(float f1, float f2);
}