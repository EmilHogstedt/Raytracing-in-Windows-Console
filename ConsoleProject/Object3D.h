#pragma once
#include "MyMath.h"

//Barebones baseclass in order to be able to group all objects together.
class Object3D
{
public:
	Object3D() : m_middlePos{ Vector3() }, m_tag { "" }
	{
	}
	Object3D(Vector3 middle, std::string tag) :
		m_middlePos{ middle }, m_tag{ tag }
	{
	}
	virtual ~Object3D() noexcept = default;

	virtual void Update(long double) = 0;
	std::string GetTag();
	Vector3 GetPos();
protected:
	Vector3 m_middlePos;
private:
	std::string m_tag;
};