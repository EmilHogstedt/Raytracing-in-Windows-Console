#include "pch.h"
#include "Object3D.h"

std::string Object3D::GetTag()
{
	return m_tag;
}

Vector3 Object3D::GetPos()
{
	return m_middlePos;
}