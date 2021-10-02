#pragma once
#include "pch.h"

//Barebones baseclass to be able to group all objects together.
class Object
{
public:
	Object() : m_middlePos{ 0.0f, 0.0f, 0.0f }, m_tag { "" }
	{
	}
	virtual ~Object() = default;
private:
	float m_middlePos[3];
	std::string m_tag;
};