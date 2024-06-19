#include "pch.h"
#include "Timer.h"

Time::Time() : m_start{ t_clock::now() }
{
	m_loopBegin/*Rendering*/ = Time::t_clock::now();
	//m_loopBeginPrinting = Time::t_clock::now();
	m_deltaTime/*Rendering*/ = Time::m_loopBegin/*Rendering*/ - Time::m_loopBegin/*Rendering*/;
	//m_deltaTimePrinting = Time::m_loopBeginPrinting - Time::m_loopBeginPrinting;
}

long double Time::SinceStart()
{
	t_dSec elapsed = t_clock::now() - m_start;
	return elapsed.count();
}

long double Time::DeltaTime()// Rendering()
{
	return m_deltaTime/*Rendering*/.count();
}
/*
long double Time::DeltaTimePrinting()
{
	return m_deltaTimePrinting.count();
}
*/
void Time::Update()//Rendering()
{
	m_deltaTime/*Rendering*/ = t_clock::now() - m_loopBegin/*Rendering*/;
	m_loopBegin/*Rendering*/ = t_clock::now();
}
/*
void Time::UpdatePrinting()
{
	m_deltaTimePrinting = t_clock::now() - m_loopBeginPrinting;
	m_loopBeginPrinting = t_clock::now();
}*/