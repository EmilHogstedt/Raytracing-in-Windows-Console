#include "pch.h"
#include "Object3D.h"

__host__ __device__ ObjectType Object3D::GetType()
{
	return m_type;
}

__host__ __device__ Vector3 Object3D::GetPos() const
{
	return m_middlePos;
}

__host__ __device__ Vector3 Object3D::GetColor()
{
	return m_color;
}

void Object3D::SetType(ObjectType type)
{
	m_type = type;
}

void Object3D::SetMiddlePos(Vector3 middlePos)
{
	m_middlePos = middlePos;
}

__device__ void AddGridCell(Vector3 id)
{
	m_GridCells.push_back(id);
}