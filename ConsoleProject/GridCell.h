#pragma once
#define __CUDACC__
#include "device_atomic_functions.h"
#include "Object3D.h"

class GridCell {
public:
	GridCell(int idx, int idy, int idz) noexcept;
	~GridCell() noexcept = default;

	__device__ void AddObjectToGridCell(Object3D* objectToAdd);
private:
	Vector3 m_id;
	int m_currentIndex = 0;
	Object3D* m_CurrentObjects[100] = { 0 };
};