#include "pch.h"
#include "Camera.h"
//#include "PrintMachine.h" Fuck off linker errors

Camera::Camera() :
	m_pos{ Vector3() },
	m_rot{ Vector3() },
	m_right{ Vector3(-1.0f, 0.0f, 0.0f) },
	m_up{ Vector3(0.0f, 1.0f, 0.0f) },
	m_forward{ Vector3(0.0f, 0.0f, 1.0f) },
	m_wMatrix{ Matrix() },
	m_vMatrix{ Matrix() },
	m_pMatrix{ Matrix() }
{
}

Camera::~Camera()
{
}

void Camera::Init()
{
	float currentFOV = M_PI / m_FOV;
	float width = 50;// (float)(PrintMachine::GetInstance()->GetWidth());
	float height = 50;// (float)(PrintMachine::GetInstance()->GetHeight());
	float aspect = width / height;

	float e = 1.0f / (std::tan(currentFOV / 2.0f));

	//Maybe have to transpose?
	//First row
	m_pMatrix.row1.x = e / aspect;
	m_pMatrix.row1.y = 0.0f;
	m_pMatrix.row1.z = 0.0f;
	m_pMatrix.row1.w = 0.0f;
	//Second row
	m_pMatrix.row2.x = 0.0f;
	m_pMatrix.row2.y = e;
	m_pMatrix.row2.z = 0.0f;
	m_pMatrix.row2.w = 0.0f;
	//Third row
	m_pMatrix.row3.x = 0.0f;
	m_pMatrix.row3.y = 0.0f;
	m_pMatrix.row3.z = (m_screenFar + m_screenNear) / (m_screenNear - m_screenFar);
	m_pMatrix.row3.w = (2 * m_screenFar * m_screenNear) / (m_screenNear - m_screenFar);
	//Fourth row
	m_pMatrix.row4.x = 0.0f;
	m_pMatrix.row4.y = 0.0f;
	m_pMatrix.row4.z = -1.0f;
	m_pMatrix.row4.w = 0.0f;
}

void Camera::Update()
{
	//Update view matrix every frame.
	//First row
	m_vMatrix.row1.x = m_right.x;
	m_vMatrix.row1.y = m_up.x;
	m_vMatrix.row1.z = m_forward.x;
	m_vMatrix.row1.w = m_pos.x;
	//Second row
	m_vMatrix.row2.x = m_right.y;
	m_vMatrix.row2.y = m_up.y;
	m_vMatrix.row2.z = m_forward.y;
	m_vMatrix.row2.w = m_pos.y;
	//Third row
	m_vMatrix.row3.x = m_right.z;
	m_vMatrix.row3.y = m_up.z;
	m_vMatrix.row3.z = m_forward.z;
	m_vMatrix.row3.w = m_pos.z;
	//Fourth row
	m_vMatrix.row4.x = 0.0f;
	m_vMatrix.row4.y = 0.0f;
	m_vMatrix.row4.z = 0.0f;
	m_vMatrix.row4.w = 1.0f;
}

void Camera::SetRot(float p, float r, float y)
{
	m_rot.x = p;
	m_rot.y = r;
	m_rot.z = y;
}

void Camera::SetPos(float x, float y, float z)
{
	m_pos.x = x;
	m_pos.y = y;
	m_pos.z = z;
}

Matrix Camera::GetWMatrix()
{
	return m_wMatrix;
}

Matrix Camera::GetVMatrix()
{
	return m_vMatrix;
}

//Doesnt work. special det is 0?
Matrix Camera::GetInverseVMatrix()
{
	Matrix InverseMatrix = Matrix();

	float det =
		m_vMatrix.row1.x * m_vMatrix.row2.y * m_vMatrix.row3.z -
		m_vMatrix.row1.x * m_vMatrix.row2.z * m_vMatrix.row3.y +
		m_vMatrix.row1.y * m_vMatrix.row2.z * m_vMatrix.row3.x +
		m_vMatrix.row1.z * m_vMatrix.row2.x * m_vMatrix.row3.y -
		m_vMatrix.row1.y * m_vMatrix.row2.x * m_vMatrix.row3.z -
		m_vMatrix.row1.z * m_vMatrix.row2.y * m_vMatrix.row3.x;

	InverseMatrix.row1.x = (m_vMatrix.row2.y * m_vMatrix.row3.z - m_vMatrix.row2.z * m_vMatrix.row3.y) / det;
	InverseMatrix.row1.y = (m_vMatrix.row1.z * m_vMatrix.row3.y - m_vMatrix.row1.y * m_vMatrix.row3.z) / det;
	InverseMatrix.row1.z = (m_vMatrix.row1.y * m_vMatrix.row2.z - m_vMatrix.row1.z * m_vMatrix.row2.y) / det;

	InverseMatrix.row2.x = (m_vMatrix.row2.z * m_vMatrix.row3.x - m_vMatrix.row2.x * m_vMatrix.row3.z) / det;
	InverseMatrix.row2.y = -(m_vMatrix.row1.z * m_vMatrix.row3.x - m_vMatrix.row1.x * m_vMatrix.row3.z) / det;
	InverseMatrix.row2.z = -(m_vMatrix.row1.x * m_vMatrix.row2.z - m_vMatrix.row1.z * m_vMatrix.row2.x) / det;

	InverseMatrix.row3.x = -(m_vMatrix.row2.y * m_vMatrix.row3.x - m_vMatrix.row2.x * m_vMatrix.row3.y) / det;
	InverseMatrix.row3.y = (m_vMatrix.row1.y * m_vMatrix.row3.x - m_vMatrix.row1.x * m_vMatrix.row3.y) / det;
	InverseMatrix.row3.z = (m_vMatrix.row1.x * m_vMatrix.row2.y - m_vMatrix.row1.y * m_vMatrix.row2.x) / det;

	InverseMatrix.row4.x = 0.0f;
	InverseMatrix.row4.y = 0.0f;
	InverseMatrix.row4.z = 0.0f;
	InverseMatrix.row4.w = 1.0f;

	//Special w values.
	float specialDet = m_vMatrix.row1.y * m_vMatrix.row3.x - m_vMatrix.row1.x * m_vMatrix.row3.y;
	InverseMatrix.row1.w = (
		m_vMatrix.row1.y * m_vMatrix.row1.w * m_vMatrix.row2.z * m_vMatrix.row3.x * m_vMatrix.row3.y -
		m_vMatrix.row1.y * m_vMatrix.row1.z * m_vMatrix.row2.w * m_vMatrix.row3.x * m_vMatrix.row3.y +
		m_vMatrix.row1.y * m_vMatrix.row1.y * m_vMatrix.row2.w * m_vMatrix.row3.x * m_vMatrix.row3.z -
		m_vMatrix.row1.y * m_vMatrix.row1.w * m_vMatrix.row2.y * m_vMatrix.row3.x * m_vMatrix.row3.z -
		m_vMatrix.row1.y * m_vMatrix.row1.y * m_vMatrix.row2.z * m_vMatrix.row3.x * m_vMatrix.row3.w +
		m_vMatrix.row1.y * m_vMatrix.row1.z * m_vMatrix.row2.y * m_vMatrix.row3.x * m_vMatrix.row3.w -
		m_vMatrix.row1.x * m_vMatrix.row1.w * m_vMatrix.row2.z * m_vMatrix.row3.y * m_vMatrix.row3.y +
		m_vMatrix.row1.x * m_vMatrix.row1.z * m_vMatrix.row2.w * m_vMatrix.row3.y * m_vMatrix.row3.y -
		m_vMatrix.row1.x * m_vMatrix.row1.y * m_vMatrix.row2.w * m_vMatrix.row3.y * m_vMatrix.row3.z +
		m_vMatrix.row1.x * m_vMatrix.row1.w * m_vMatrix.row2.y * m_vMatrix.row3.y * m_vMatrix.row3.z +
		m_vMatrix.row1.x * m_vMatrix.row1.y * m_vMatrix.row2.z * m_vMatrix.row3.y * m_vMatrix.row3.w -
		m_vMatrix.row1.x * m_vMatrix.row1.z * m_vMatrix.row2.y * m_vMatrix.row3.y * m_vMatrix.row3.w
		)
		/ (specialDet * det);

	InverseMatrix.row2.w = (
		m_vMatrix.row1.x * m_vMatrix.row1.x * m_vMatrix.row2.w * m_vMatrix.row3.y * m_vMatrix.row3.z -
		m_vMatrix.row1.x * m_vMatrix.row1.x * m_vMatrix.row2.z * m_vMatrix.row3.y * m_vMatrix.row3.w +
		m_vMatrix.row1.x * m_vMatrix.row1.w * m_vMatrix.row2.z * m_vMatrix.row3.x * m_vMatrix.row3.y -
		m_vMatrix.row1.x * m_vMatrix.row1.z * m_vMatrix.row2.w * m_vMatrix.row3.x * m_vMatrix.row3.y -
		m_vMatrix.row1.x * m_vMatrix.row1.y * m_vMatrix.row2.w * m_vMatrix.row3.x * m_vMatrix.row3.z -
		m_vMatrix.row1.x * m_vMatrix.row1.w * m_vMatrix.row2.x * m_vMatrix.row3.y * m_vMatrix.row3.z +
		m_vMatrix.row1.x * m_vMatrix.row1.y * m_vMatrix.row2.z * m_vMatrix.row3.x * m_vMatrix.row3.w +
		m_vMatrix.row1.x * m_vMatrix.row1.z * m_vMatrix.row2.x * m_vMatrix.row3.y * m_vMatrix.row3.w -
		m_vMatrix.row1.y * m_vMatrix.row1.w * m_vMatrix.row2.z * m_vMatrix.row3.x * m_vMatrix.row3.x +
		m_vMatrix.row1.y * m_vMatrix.row1.z * m_vMatrix.row2.w * m_vMatrix.row3.x * m_vMatrix.row3.x +
		m_vMatrix.row1.y * m_vMatrix.row1.w * m_vMatrix.row2.x * m_vMatrix.row3.x * m_vMatrix.row3.z -
		m_vMatrix.row1.y * m_vMatrix.row1.z * m_vMatrix.row2.x * m_vMatrix.row3.x * m_vMatrix.row3.w
		)
		/ (specialDet * det);

	InverseMatrix.row3.w = -(
		m_vMatrix.row1.x * m_vMatrix.row2.y * m_vMatrix.row3.w -
		m_vMatrix.row1.x * m_vMatrix.row2.w * m_vMatrix.row3.y +
		m_vMatrix.row1.y * m_vMatrix.row2.w * m_vMatrix.row3.x +
		m_vMatrix.row1.w * m_vMatrix.row2.x * m_vMatrix.row3.y -
		m_vMatrix.row1.y * m_vMatrix.row2.x * m_vMatrix.row3.w -
		m_vMatrix.row1.w * m_vMatrix.row2.y * m_vMatrix.row3.x
		) / det;

	return InverseMatrix;
}

Matrix Camera::GetPMatrix()
{
	return m_pMatrix;
}

Vector3 Camera::GetPos()
{
	return m_pos;
}

Vector3 Camera::GetRot()
{
	return m_rot;
}
