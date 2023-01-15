#include "pch.h"
#include "GridCell.h"

GridCell::GridCell(int idx, int idy, int idz) noexcept
{
	m_id = Vector3(idx, idy, idz);
}

__device__ void GridCell::AddObjectToGridCell(Object3D* objectToAdd)
{
	unsigned int index = atomicAdd(&m_currentObjectIndex, 1);
	m_currentObjects[index] = objectToAdd;
	return;
}

__device__ void GridCell::AddPointLightToGridCell(PointLight* pointLightToAdd)
{
	unsigned int index = atomicAdd(&m_currentPointLightIndex, 1);
	m_currentPointLights[index] = pointLightToAdd;
	return;
}

__device__ int GridCell::GetObjectCount() const noexcept
{
	return m_currentObjectIndex;
}

__device__ int GridCell::GetPointLightCount() const noexcept
{
	return m_currentPointLightIndex;
}

__device__ Object3D* GridCell::GetCellObject(int id) const noexcept
{
	return m_currentObjects[id];
}

__device__ PointLight* GridCell::GetCellPointLight(int id) const noexcept
{
	return m_currentPointLights[id];
}

__device__ void GridCell::Reset()
{
	m_currentObjectIndex = 0;
	m_currentPointLightIndex = 0;
	memset(m_currentObjects, 0, sizeof(Object3D*) * GRID_CAPACITY);
	memset(m_currentPointLights, 0, sizeof(PointLight*) * GRID_CAPACITY);
}