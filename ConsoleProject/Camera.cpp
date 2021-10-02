#include "pch.h"
#include "Camera.h"
#include "PrintMachine.h"

Camera::Camera() :
	m_pos{ 0.0f, 0.0f, 0.0f },
	m_rot{ 0.0f, 0.0f, 0.0f },
	m_right{ 1.0f, 0.0f, 0.0f },
	m_up{ 0.0f, 1.0f, 0.0f },
	m_forward{ 0.0f, 0.0f, 1.0f }
{
	m_wMatrix = DBG_NEW float*[4];
	m_vMatrix = DBG_NEW float*[4];
	m_pMatrix = DBG_NEW float*[4];
	for (size_t i = 0; i < 4; i++)
	{
		m_wMatrix[i] = DBG_NEW float[4];
		m_vMatrix[i] = DBG_NEW float[4];
		m_pMatrix[i] = DBG_NEW float[4];
		for (size_t j = 0; j < 4; j++)
		{
			m_wMatrix[i][j] = 0.0f;
			m_vMatrix[i][j] = 0.0f;
			m_pMatrix[i][j] = 0.0f;
		}
	}
}

Camera::~Camera()
{
	for (size_t i = 0; i < 4; i++)
	{
		delete[] m_wMatrix[i];
		delete[] m_vMatrix[i];
		delete[] m_pMatrix[i];
	}
	delete[] m_wMatrix;
	delete[] m_vMatrix;
	delete[] m_pMatrix;
}

void Camera::Init()
{
	float currentFOV = M_PI / m_FOV;
	float width = (float)(PrintMachine::GetInstance()->GetWidth());
	float height = (float)(PrintMachine::GetInstance()->GetHeight());
	float aspect = width / height;

	float e = 1.0f / (std::tan(currentFOV / 2.0f));	//Maybe should be m_FOV instead of currentFOV?

	//Maybe have to transpose?
	//First row
	m_pMatrix[0][0] = e / aspect;
	m_pMatrix[0][1] = 0.0f;
	m_pMatrix[0][2] = 0.0f;
	m_pMatrix[0][3] = 0.0f;
	//Second row
	m_pMatrix[1][0] = 0.0f;
	m_pMatrix[1][1] = e;
	m_pMatrix[1][2] = 0.0f;
	m_pMatrix[1][3] = 0.0f;
	//Third row
	m_pMatrix[2][0] = 0.0f;
	m_pMatrix[2][1] = 0.0f;
	m_pMatrix[2][2] = (m_screenFar + m_screenNear) / (m_screenNear - m_screenFar);
	m_pMatrix[2][3] = (2 * m_screenFar * m_screenNear) / (m_screenNear - m_screenFar);
	//Fourth row
	m_pMatrix[3][0] = 0.0f;
	m_pMatrix[3][1] = 0.0f;
	m_pMatrix[3][2] = -1.0f;
	m_pMatrix[3][3] = 0.0f;
}

void Camera::Update()
{
	//Update view matrix every frame.
	//First row
	m_vMatrix[0][0] = m_right[0];
	m_vMatrix[0][1] = m_up[0];
	m_vMatrix[0][2] = m_forward[0];
	m_vMatrix[0][3] = m_pos[0];
	//Second row
	m_vMatrix[1][0] = m_right[1];
	m_vMatrix[1][1] = m_up[1];
	m_vMatrix[1][2] = m_forward[1];
	m_vMatrix[1][3] = m_pos[1];
	//Third row
	m_vMatrix[2][0] = m_right[2];
	m_vMatrix[2][1] = m_up[2];
	m_vMatrix[2][2] = m_forward[2];
	m_vMatrix[2][3] = m_pos[2];
	//Fourth row
	m_vMatrix[3][0] = 0.0f;
	m_vMatrix[3][1] = 0.0f;
	m_vMatrix[3][2] = 0.0f;
	m_vMatrix[3][3] = 1.0f;
}

void Camera::SetRot(float p, float r, float y)
{
	m_rot[0] = p;
	m_rot[1] = r;
	m_rot[2] = y;
}

void Camera::SetPos(float x, float y, float z)
{
	m_pos[0] = x;
	m_pos[1] = y;
	m_pos[2] = z;
}

float** Camera::GetWMatrix()
{
	return m_wMatrix;
}

float** Camera::GetVMatrix()
{
	return m_vMatrix;
}

float** Camera::GetPMatrix()
{
	return m_pMatrix;
}

float* Camera::GetPos()
{
	return m_pos;
}

float* Camera::GetRot()
{
	return m_rot;
}