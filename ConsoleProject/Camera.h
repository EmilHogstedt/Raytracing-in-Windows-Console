#pragma once

class Camera
{
public:
	Camera();
	~Camera();

	void Init();
	void Update();
	void SetRot(float, float, float);
	void SetPos(float, float, float);

	float** GetWMatrix();
	float** GetVMatrix();
	float** GetPMatrix();

	float* GetPos();
	float* GetRot();
private:
	float** m_wMatrix;
	float** m_vMatrix;
	float** m_pMatrix;

	float m_right[3];
	float m_up[3];
	float m_forward[3];

	float m_pos[3];
	float m_rot[3];

	float m_screenNear = 0.1f;
	float m_screenFar = 10000.0f;

	float m_FOV = 4.0f;
};