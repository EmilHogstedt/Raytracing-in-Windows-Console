#include "pch.h"
#include "GridCell.h"

GridCell::GridCell(int idx, int idy, int idz) noexcept
{
	m_id = Vector3(idx, idy, idz);
}

__device__ void GridCell::AddObjectToGridCell(Object3D* objectToAdd)
{
	m_CurrentObjects[atomicAdd(&m_currentIndex, 1)] = objectToAdd;
	objectToAdd->AddGridCell(this);
	return;
}
