#include "pch.h"
#include "Camera3D.h"
#include "PrintMachine.h"

//The pMatrix is set up in init. Since there is no option to change it atm.
void Camera3D::Init()
{
	float currentFOV = (float)(M_PI) / m_FOV;
	float width = (float)(PrintMachine::GetWidth());
	float height = (float)(PrintMachine::GetHeight());

	//Increase the value to to squish x, decrease to drag out x
	//Set it to 0.005f for high resolution.
	//Set it to 0.01f for low resolution.
	float aspect = width / (0.01f * width * height);

	float e = 1.0f / (tan(currentFOV / 2.0f));

	m_hNear = 2.0f * tan(currentFOV / 2.0f) * m_screenNear;
	m_wNear = m_hNear * aspect;

	m_hFar = 2.0f * tan(currentFOV / 2.0f) * m_screenFar;
	m_wFar = m_hFar * aspect;

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
	m_pMatrix.row3.w = (2.0f * m_screenFar * m_screenNear) / (m_screenNear - m_screenFar);
	//Fourth row
	m_pMatrix.row4.x = 0.0f;
	m_pMatrix.row4.y = 0.0f;
	m_pMatrix.row4.z = -1.0f;
	m_pMatrix.row4.w = 0.0f;
}

//All the vectors and the vMatrix gets updated using the camera's movement and rotation.
void Camera3D::Update()
{
	float p = m_rot.x;
	float y = m_rot.y;
	float r = m_rot.z;
	
	m_forward.x = -sin(y);
	m_forward.y = -sin(p) * cos(y);
	m_forward.z = -cos(p) * cos(y);

	m_staticForward.x = -sin(y);
	m_staticForward.y = -cos(y);
	m_staticForward.z = -cos(y);

	m_right.x = cos(y);
	m_right.y = -sin(p) * sin(y);
	m_right.z = -cos(p) * sin(y);

	m_staticRight.x = cos(y);
	m_staticRight.y = -sin(y);
	m_staticRight.z = -sin(y);

	m_up.x = 0.0f;
	m_up.y = cos(p);
	m_up.z = -sin(p);
	
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

const float Camera3D::GetFarPlaneDistance() const
{
	return m_screenFar;
}

const MyMath::Vector4& Camera3D::GetFrustum() const
{
	return MyMath::Vector4(m_wNear, m_hNear, m_wFar, m_hFar);
}

const MyMath::Vector3& Camera3D::GetRight() const
{
	return m_right;
}

const MyMath::Vector3& Camera3D::GetUp() const
{
	return m_up;
}

const MyMath::Vector3& Camera3D::GetForward() const
{
	return m_forward;
}

//Set rot to a specific value, when teleporting.
void Camera3D::SetRot(const float p, const float y, const float r)
{
	m_rot.x = p;
	m_rot.y = y;
	m_rot.z = r;
}

//Set pos to a specific value, when teleporting.
void Camera3D::SetPos(const float x, const float y, const float z)
{
	m_pos.x = x;
	m_pos.y = y;
	m_pos.z = z;
}

//Moves using the keymap.
void Camera3D::Move(const long double dt)
{
	static const float speed = 10.0f;

	const float deltaSpeed = dt * speed;

	//#todo: Remake this. Right now it moves faster diagonally. Should also use velocity. 
	
	//Use the current right & forward vectors to calculate movement in the x-z plane.
	MyMath::Vector3 moveX = m_staticRight * static_cast<float>(m_Keys.D - m_Keys.A);
	MyMath::Vector3 moveZ = m_staticForward * static_cast<float>(m_Keys.W - m_Keys.S);

	MyMath::Vector3 totalMovement = moveX + moveZ;
	totalMovement.Normalize_InPlace();

	//Then add those to the current posision.
	m_pos.x = m_pos.x + (totalMovement.x * deltaSpeed);// +(moveZ.x * deltaSpeed);
	m_pos.z = m_pos.z + (totalMovement.z * deltaSpeed);// +(moveZ.z * deltaSpeed);
	
	//The updating of the y-pos does not get effected by the rotation of the axis'
	m_pos.y += (m_Keys.Space - m_Keys.Shift) * deltaSpeed;
}

//Used to add rotation to the already existing rotational position of the camera when moving the mouse, for example.
void Camera3D::AddRot(const long double dt, const short p, const short y, const short r)
{
	static const float speed = 0.002f;
	
	//For now we do not want to use DT for camera rotations, as the speed increases when the FPS decreases.
	//We want the same amount of rotation to occur no matter how high/low the fps is when the mouse is moved a certain distance this frame.
	const float deltaSpeed = /*dt **/ speed;

	m_rot.x -= ((float)p * deltaSpeed);
	m_rot.y += ((float)y * deltaSpeed);
	m_rot.z += ((float)r * deltaSpeed);

	if (m_rot.x > static_cast<float>(M_PI / 2.0))
	{
		m_rot.x = static_cast<float>((M_PI / 2.0) - 0.0001);
	}

	if (m_rot.x < static_cast<float>(-M_PI / 2.0))
	{
		m_rot.x = static_cast<float>((-M_PI / 2.0) + 0.0001);
	}
}

//TODO: Move MouseCoords to UI/2D manager.
const COORD& Camera3D::GetMouseCoords()
{
	return m_mouseCoords;
}

void Camera3D::SetMouseCoords(const COORD& newCoords)
{
	m_mouseCoords = newCoords;
}

const MyMath::Matrix& Camera3D::GetVMatrix() const
{
	return m_vMatrix;
}

//Manual calculation of the inverse 4x4 matrix. Is there any way to optimize this???
//Here be dragons
const MyMath::Matrix& Camera3D::GetInverseVMatrix() const
{
	MyMath::Matrix inverseMatrix = MyMath::Matrix();

	//First column.
	//00 Done
	inverseMatrix.row1.x = 
		m_vMatrix.row2.y * m_vMatrix.row3.z * m_vMatrix.row4.w -
		m_vMatrix.row2.y * m_vMatrix.row3.w * m_vMatrix.row4.z -
		m_vMatrix.row3.y * m_vMatrix.row2.z * m_vMatrix.row4.w +
		m_vMatrix.row3.y * m_vMatrix.row2.w * m_vMatrix.row4.z +
		m_vMatrix.row4.y * m_vMatrix.row2.z * m_vMatrix.row3.w -
		m_vMatrix.row4.y * m_vMatrix.row2.w * m_vMatrix.row3.z;

	//10 Done
	inverseMatrix.row2.x =
		-m_vMatrix.row2.x * m_vMatrix.row3.z * m_vMatrix.row4.w +
		m_vMatrix.row2.x * m_vMatrix.row3.w * m_vMatrix.row4.z +
		m_vMatrix.row3.x * m_vMatrix.row2.z * m_vMatrix.row4.w -
		m_vMatrix.row3.x * m_vMatrix.row2.w * m_vMatrix.row4.z -
		m_vMatrix.row4.x * m_vMatrix.row2.z * m_vMatrix.row3.w +
		m_vMatrix.row4.x * m_vMatrix.row2.w * m_vMatrix.row3.z;

	//20 Done
	inverseMatrix.row3.x =
		m_vMatrix.row2.x * m_vMatrix.row3.y * m_vMatrix.row4.w -
		m_vMatrix.row2.x * m_vMatrix.row3.w * m_vMatrix.row4.y -
		m_vMatrix.row3.x * m_vMatrix.row2.y * m_vMatrix.row4.w +
		m_vMatrix.row3.x * m_vMatrix.row2.w * m_vMatrix.row4.y +
		m_vMatrix.row4.x * m_vMatrix.row2.y * m_vMatrix.row3.w -
		m_vMatrix.row4.x * m_vMatrix.row2.w * m_vMatrix.row3.y;

	//30 Done
	inverseMatrix.row4.x =
		-m_vMatrix.row2.x * m_vMatrix.row3.y * m_vMatrix.row4.z +
		m_vMatrix.row2.x * m_vMatrix.row3.z * m_vMatrix.row4.y +
		m_vMatrix.row3.x * m_vMatrix.row2.y * m_vMatrix.row4.z -
		m_vMatrix.row3.x * m_vMatrix.row2.z * m_vMatrix.row4.y -
		m_vMatrix.row4.x * m_vMatrix.row2.y * m_vMatrix.row3.z +
		m_vMatrix.row4.x * m_vMatrix.row2.z * m_vMatrix.row3.y;

	//Second column.
	//01 Done
	inverseMatrix.row1.y =
		-m_vMatrix.row1.y * m_vMatrix.row3.z * m_vMatrix.row4.w +
		m_vMatrix.row1.y * m_vMatrix.row3.w * m_vMatrix.row4.z +
		m_vMatrix.row3.y * m_vMatrix.row1.z * m_vMatrix.row4.w -
		m_vMatrix.row3.y * m_vMatrix.row1.w * m_vMatrix.row4.z -
		m_vMatrix.row4.y * m_vMatrix.row1.z * m_vMatrix.row3.w +
		m_vMatrix.row4.y * m_vMatrix.row1.w * m_vMatrix.row3.z;

	//11 Done
	inverseMatrix.row2.y =
		m_vMatrix.row1.x * m_vMatrix.row3.z * m_vMatrix.row4.w -
		m_vMatrix.row1.x * m_vMatrix.row3.w * m_vMatrix.row4.z -
		m_vMatrix.row3.x * m_vMatrix.row1.z * m_vMatrix.row4.w +
		m_vMatrix.row3.x * m_vMatrix.row1.w * m_vMatrix.row4.z +
		m_vMatrix.row4.x * m_vMatrix.row1.z * m_vMatrix.row3.w -
		m_vMatrix.row4.x * m_vMatrix.row1.w * m_vMatrix.row3.z;

	//21 Done
	inverseMatrix.row3.y =
		-m_vMatrix.row1.x * m_vMatrix.row3.y * m_vMatrix.row4.w +
		m_vMatrix.row1.x * m_vMatrix.row3.w * m_vMatrix.row4.y +
		m_vMatrix.row3.x * m_vMatrix.row1.y * m_vMatrix.row4.w -
		m_vMatrix.row3.x * m_vMatrix.row1.w * m_vMatrix.row4.y -
		m_vMatrix.row4.x * m_vMatrix.row1.y * m_vMatrix.row3.w +
		m_vMatrix.row4.x * m_vMatrix.row1.w * m_vMatrix.row3.y;

	//31 Done
	inverseMatrix.row4.y =
		m_vMatrix.row1.x * m_vMatrix.row3.y * m_vMatrix.row4.z -
		m_vMatrix.row1.x * m_vMatrix.row3.z * m_vMatrix.row4.y -
		m_vMatrix.row3.x * m_vMatrix.row1.y * m_vMatrix.row4.z +
		m_vMatrix.row3.x * m_vMatrix.row1.z * m_vMatrix.row4.y +
		m_vMatrix.row4.x * m_vMatrix.row1.y * m_vMatrix.row3.z -
		m_vMatrix.row4.x * m_vMatrix.row1.z * m_vMatrix.row3.y;

	//Third column.
	//02 Done
	inverseMatrix.row1.z =
		m_vMatrix.row1.y * m_vMatrix.row2.z * m_vMatrix.row4.w -
		m_vMatrix.row1.y * m_vMatrix.row2.w * m_vMatrix.row4.z -
		m_vMatrix.row2.y * m_vMatrix.row1.z * m_vMatrix.row4.w +
		m_vMatrix.row2.y * m_vMatrix.row1.w * m_vMatrix.row4.z +
		m_vMatrix.row4.y * m_vMatrix.row1.z * m_vMatrix.row2.w -
		m_vMatrix.row4.y * m_vMatrix.row1.w * m_vMatrix.row2.z;

	//12 Done
	inverseMatrix.row2.z =
		-m_vMatrix.row1.x * m_vMatrix.row2.z * m_vMatrix.row4.w +
		m_vMatrix.row1.x * m_vMatrix.row2.w * m_vMatrix.row4.z +
		m_vMatrix.row2.x * m_vMatrix.row1.z * m_vMatrix.row4.w -
		m_vMatrix.row2.x * m_vMatrix.row1.w * m_vMatrix.row4.z -
		m_vMatrix.row4.x * m_vMatrix.row1.z * m_vMatrix.row2.w +
		m_vMatrix.row4.x * m_vMatrix.row1.w * m_vMatrix.row2.z;

	//22 Done
	inverseMatrix.row3.z =
		m_vMatrix.row1.x * m_vMatrix.row2.y * m_vMatrix.row4.w -
		m_vMatrix.row1.x * m_vMatrix.row2.w * m_vMatrix.row4.y -
		m_vMatrix.row2.x * m_vMatrix.row1.y * m_vMatrix.row4.w +
		m_vMatrix.row2.x * m_vMatrix.row1.w * m_vMatrix.row4.y +
		m_vMatrix.row4.x * m_vMatrix.row1.y * m_vMatrix.row2.w -
		m_vMatrix.row4.x * m_vMatrix.row1.w * m_vMatrix.row2.y;

	//32 Done
	inverseMatrix.row4.z =
		-m_vMatrix.row1.x * m_vMatrix.row2.y * m_vMatrix.row4.z +
		m_vMatrix.row1.x * m_vMatrix.row2.z * m_vMatrix.row4.y +
		m_vMatrix.row2.x * m_vMatrix.row1.y * m_vMatrix.row4.z -
		m_vMatrix.row2.x * m_vMatrix.row1.z * m_vMatrix.row4.y -
		m_vMatrix.row4.x * m_vMatrix.row1.y * m_vMatrix.row2.z +
		m_vMatrix.row4.x * m_vMatrix.row1.z * m_vMatrix.row2.y;

	//Fourth column.
	//03 Done
	inverseMatrix.row1.w =
		-m_vMatrix.row1.y * m_vMatrix.row2.z * m_vMatrix.row3.w +
		m_vMatrix.row1.y * m_vMatrix.row2.w * m_vMatrix.row3.z +
		m_vMatrix.row2.y * m_vMatrix.row1.z * m_vMatrix.row3.w -
		m_vMatrix.row2.y * m_vMatrix.row1.w * m_vMatrix.row3.z -
		m_vMatrix.row3.y * m_vMatrix.row1.z * m_vMatrix.row2.w +
		m_vMatrix.row3.y * m_vMatrix.row1.w * m_vMatrix.row2.z;

	//13 Done
	inverseMatrix.row2.w =
		m_vMatrix.row1.x * m_vMatrix.row2.z * m_vMatrix.row3.w -
		m_vMatrix.row1.x * m_vMatrix.row2.w * m_vMatrix.row3.z -
		m_vMatrix.row2.x * m_vMatrix.row1.z * m_vMatrix.row3.w +
		m_vMatrix.row2.x * m_vMatrix.row1.w * m_vMatrix.row3.z +
		m_vMatrix.row3.x * m_vMatrix.row1.z * m_vMatrix.row2.w -
		m_vMatrix.row3.x * m_vMatrix.row1.w * m_vMatrix.row2.z;

	//23 Done
	inverseMatrix.row3.w =
		-m_vMatrix.row1.x * m_vMatrix.row2.y * m_vMatrix.row3.w +
		m_vMatrix.row1.x * m_vMatrix.row2.w * m_vMatrix.row3.y +
		m_vMatrix.row2.x * m_vMatrix.row1.y * m_vMatrix.row3.w -
		m_vMatrix.row2.x * m_vMatrix.row1.w * m_vMatrix.row3.y -
		m_vMatrix.row3.x * m_vMatrix.row1.y * m_vMatrix.row2.w +
		m_vMatrix.row3.x * m_vMatrix.row1.w * m_vMatrix.row2.y;

	//33 Done
	inverseMatrix.row4.w =
		m_vMatrix.row1.x * m_vMatrix.row2.y * m_vMatrix.row3.z -
		m_vMatrix.row1.x * m_vMatrix.row2.z * m_vMatrix.row3.y -
		m_vMatrix.row2.x * m_vMatrix.row1.y * m_vMatrix.row3.z +
		m_vMatrix.row2.x * m_vMatrix.row1.z * m_vMatrix.row3.y +
		m_vMatrix.row3.x * m_vMatrix.row1.y * m_vMatrix.row2.z -
		m_vMatrix.row3.x * m_vMatrix.row1.z * m_vMatrix.row2.y;

	float det =
		m_vMatrix.row1.x * inverseMatrix.row1.x +
		m_vMatrix.row1.y * inverseMatrix.row2.x +
		m_vMatrix.row1.z * inverseMatrix.row3.x +
		m_vMatrix.row1.w * inverseMatrix.row4.x;

	if (det == 0.0f)
	{
		assert(false && "Determinant was 0 when inversing the view matrix.");
	}

	det = 1.0f / det;
	inverseMatrix.row1.x *= det;
	inverseMatrix.row1.y *= det;
	inverseMatrix.row1.z *= det;
	inverseMatrix.row1.w *= det;

	inverseMatrix.row2.x *= det;
	inverseMatrix.row2.y *= det;
	inverseMatrix.row2.z *= det;
	inverseMatrix.row2.w *= det;

	inverseMatrix.row3.x *= det;
	inverseMatrix.row3.y *= det;
	inverseMatrix.row3.z *= det;
	inverseMatrix.row3.w *= det;

	inverseMatrix.row4.x *= det;
	inverseMatrix.row4.y *= det;
	inverseMatrix.row4.z *= det;
	inverseMatrix.row4.w *= det;

	return inverseMatrix;
}

const MyMath::Matrix& Camera3D::GetPMatrix() const
{
	return m_pMatrix;
}

const MyMath::Vector3& Camera3D::GetPos() const
{
	return m_pos;
}

const MyMath::Vector3& Camera3D::GetRot() const
{
	return m_rot;
}
