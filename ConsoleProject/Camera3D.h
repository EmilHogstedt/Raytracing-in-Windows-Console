#pragma once
#include "MyMath.h"

//A column major 3D camera.
class Camera3D
{
public:
	Camera3D();
	~Camera3D();

	void Init();
	void Update();
	void SetRot(float, float, float);
	void SetPos(float, float, float);
	void Move(long double dt);
	void AddRot(short, short, short);

	Matrix GetVMatrix();
	Matrix GetInverseVMatrix();
	Matrix GetPMatrix();

	Vector3 GetPos();
	Vector3 GetRot();

	Vector3 GetRight();
	Vector3 GetUp();
	Vector3 GetForward();

	Vector4 GetFrustum();

	COORD GetMouseCoords();
	void SetMouseCoords(COORD newCords);

	//For keeping track of what keys are currently being pressed so that the movement can be updated accordingly.
	struct PressedKeys
	{
		int W = 0;
		int A = 0;
		int S = 0;
		int D = 0;

		int Space = 0;
		int Shift = 0;
	};

	PressedKeys m_Keys;

private:
	Matrix m_vMatrix;
	Matrix m_pMatrix;

	Vector3 m_right;
	Vector3 m_up;
	Vector3 m_forward;

	//Used when moving. Are only updated using the yaw and not the pitch.
	Vector3 m_staticForward;
	Vector3 m_staticRight;

	Vector3 m_pos;
	//pitch yaw roll
	Vector3 m_rot;
	
	float m_screenNear = 0.1f;
	float m_screenFar = 10000.0f; //Lower this. Cant see anything noteworthy after ~500 anyway. 
	float m_hNear;
	float m_wNear;
	float m_hFar;
	float m_wFar;

	COORD m_mouseCoords;

	float m_FOV = 2.0f; //Do not set to 1
};