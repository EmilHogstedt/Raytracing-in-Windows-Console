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

__host__ __device__ void Object3D::AddGridCell(GridCell* cellToAdd)
{
	m_gridCells[m_currentCells] = cellToAdd;
	m_currentCells++;
}