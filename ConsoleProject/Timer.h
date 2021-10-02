#pragma once
#include "pch.h"

class Time
{
	using t_clock = std::chrono::high_resolution_clock;
	using t_dSec = std::chrono::duration<long double, std::ratio<1, 1>>;
	using t_dNano = std::chrono::duration<long double, std::ratio<1, 1000000000>>;
	using t_moment = std::chrono::time_point<t_clock, t_dNano>;
private:
	t_moment m_start;
	static t_moment m_loopBegin;
	static t_dSec m_deltaTime;
public:
	Time();
	virtual ~Time() = default;
	long double SinceStart();
	long double DeltaTime();
	void Update();
};

Time::t_moment Time::m_loopBegin = Time::t_clock::now();
Time::t_dSec Time::m_deltaTime = Time::m_loopBegin - Time::m_loopBegin;

Time::Time() : m_start{ t_clock::now() }
{

}

long double Time::SinceStart()
{
	t_dSec elapsed = t_clock::now() - m_start;
	return elapsed.count();
}

long double Time::DeltaTime()
{
	return m_deltaTime.count();
}

void Time::Update()
{
	m_deltaTime = t_clock::now() - m_loopBegin;
	m_loopBegin = t_clock::now();
}