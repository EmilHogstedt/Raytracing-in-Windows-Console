#include "pch.h"
#include "Object3D.h"

Object3D::Object3D(const MyMath::Vector3& center, const ObjectType type, const MyMath::Vector3& color)
{
	m_center = center;
	m_type = type;
	m_color = color;
}

__host__ __device__
ObjectType Object3D::GetType() const
{
	return m_type;
}

__host__ __device__
MyMath::Vector3 Object3D::GetPos() const
{
	return m_center;
}

__host__ __device__
MyMath::Vector3 Object3D::GetColor() const
{
	return m_color;
}

void Object3D::SetType(const ObjectType type)
{
	m_type = type;
}

void Object3D::SetMiddlePos(const MyMath::Vector3& center)
{
	m_center = center;
}