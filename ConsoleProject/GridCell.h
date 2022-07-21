#pragma once
#define __CUDACC__
#include "device_atomic_functions.h"
#include "Object3D.h"

class __align__(16) GridCell {
public:
	GridCell(int idx, int idy, int idz) noexcept;
	~GridCell() noexcept = default;

	__device__ void AddObjectToGridCell(Object3D* objectToAdd);
	__device__ int GetObjectCount() const noexcept;
	__device__ Object3D* GetCellObject(int id) const noexcept;
	__device__ void Reset();
private:
	Vector3 m_id;
	int m_currentIndex = 0;
	Object3D* m_currentObjects[100] = { 0 };
};