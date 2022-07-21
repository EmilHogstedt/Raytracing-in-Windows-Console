#include "pch.h"
#include "GridCell.h"

GridCell::GridCell(int idx, int idy, int idz) noexcept
{
	m_id = Vector3(idx, idy, idz);
}

__device__ void GridCell::AddObjectToGridCell(Object3D* objectToAdd)
{
	unsigned int index = atomicAdd(&m_currentIndex, 1);
	m_currentObjects[index] = objectToAdd;
	return;
}

__device__ int GridCell::GetObjectCount() const noexcept
{
	return m_currentIndex;
}

__device__ Object3D* GridCell::GetCellObject(int id) const noexcept
{
	return m_currentObjects[id];
}

__device__ void GridCell::Reset()
{
	m_currentIndex = 0;
	memset(m_currentObjects, 0, sizeof(Object3D*) * 100);
}