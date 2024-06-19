#pragma once
#include "MyMath.h"

//A column major 3D camera.
class Camera3D
{
public:
	Camera3D() = default;
	~Camera3D() = default;

	void Init();
	void Update();
	void SetRot(const float p, const float y, const float r);
	void SetPos(const float x, const float y , const float z);
	void Move(const long double dt);
	void AddRot(const long double dt, const short p, const short y, const short r);

	const MyMath::Matrix& GetVMatrix() const;
	const MyMath::Matrix& GetInverseVMatrix() const;
	const MyMath::Matrix& GetPMatrix() const;

	const MyMath::Vector3& GetPos() const;
	const MyMath::Vector3& GetRot() const;

	const MyMath::Vector3& GetRight() const;
	const MyMath::Vector3& GetUp() const;
	const MyMath::Vector3& GetForward() const;

	const float GetFarPlaneDistance() const;
	const MyMath::Vector4& GetFrustum() const;

	void SetMouseCoords(const COORD& newCords);
	const COORD& GetMouseCoords();

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
	MyMath::Matrix m_vMatrix = MyMath::Matrix();
	MyMath::Matrix m_pMatrix = MyMath::Matrix();

	MyMath::Vector3 m_right = MyMath::Vector3(-1.0f, 0.0f, 0.0f);
	MyMath::Vector3 m_up = MyMath::Vector3(0.0f, 1.0f, 0.0f);
	MyMath::Vector3 m_forward = MyMath::Vector3(0.0f, 0.0f, 1.0f);

	//Used when moving. Are only updated using the yaw and not the pitch.
	MyMath::Vector3 m_staticRight = MyMath::Vector3(-1.0f, 0.0f, 0.0f);
	MyMath::Vector3 m_staticForward = MyMath::Vector3(0.0f, 0.0f, 1.0f);

	MyMath::Vector3 m_pos = MyMath::Vector3();

	//pitch yaw roll
	MyMath::Vector3 m_rot = MyMath::Vector3(0.0f, static_cast<float>(M_PI), 0.0f);
	
	float m_hNear = 0.0f;
	float m_wNear = 0.0f;
	float m_hFar = 0.0f;
	float m_wFar = 0.0f;

	COORD m_mouseCoords = { static_cast<SHORT>(-1.0f), static_cast<SHORT>(-1.0f) };

	const float m_screenNear = 0.1f;
	const float m_screenFar = 250.0f; //Lower this. Cant see anything noteworthy after ~500 anyway. Maybe set this depending on pixel resolution? For example the height of the screen * 2.

	//Set to 1.5 for high resolution.
	//Set to 2.0 for low resolution.
	//(1.5 seems to be a good FOV for both)
	const float m_FOV = 1.5f; //Do not set to 1
};