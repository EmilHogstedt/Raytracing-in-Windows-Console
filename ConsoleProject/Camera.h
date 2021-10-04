#pragma once
#include "MyMath.h"

class Camera
{
public:
	Camera();
	~Camera();

	void Init();
	void Update();
	void SetRot(float, float, float);
	void SetPos(float, float, float);
	
	Matrix GetWMatrix();
	Matrix GetVMatrix();
	Matrix GetInverseVMatrix();
	Matrix GetPMatrix();

	Vector3 GetPos();
	Vector3 GetRot();
private:
	Matrix m_wMatrix;
	Matrix m_vMatrix;
	Matrix m_pMatrix;

	Vector3 m_right;
	Vector3 m_up;
	Vector3 m_forward;

	Vector3 m_pos;
	Vector3 m_rot;
	
	float m_screenNear = 0.1f;
	float m_screenFar = 10000.0f;

	float m_FOV = 2.0f; //Do not set to 1
};