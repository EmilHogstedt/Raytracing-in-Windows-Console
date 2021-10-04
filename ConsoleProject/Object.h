#pragma once
#include "MyMath.h"

//Barebones baseclass to be able to group all objects together.
class Object
{
public:
	Object() : m_middlePos{ Vector3() }, m_tag { "" }
	{
	}
	Object(Vector3 middle, std::string tag) :
		m_middlePos{ middle }, m_tag{ tag }
	{
	}
	virtual ~Object() noexcept = default;

	virtual void Update() = 0;
	std::string GetTag();
	Vector3 GetPos();
protected:
	Vector3 m_middlePos;
private:
	std::string m_tag;
};