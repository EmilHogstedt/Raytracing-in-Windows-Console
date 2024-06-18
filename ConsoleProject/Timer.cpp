#include "pch.h"
#include "Timer.h"

Time::Time() : m_start{ t_clock::now() }
{
	m_loopBeginRendering = Time::t_clock::now();
	m_loopBeginPrinting = Time::t_clock::now();
	m_deltaTimeRendering = Time::m_loopBeginRendering - Time::m_loopBeginRendering;
	m_deltaTimePrinting = Time::m_loopBeginPrinting - Time::m_loopBeginPrinting;
}

long double Time::SinceStart()
{
	t_dSec elapsed = t_clock::now() - m_start;
	return elapsed.count();
}

long double Time::DeltaTimeRendering()
{
	return m_deltaTimeRendering.count();
}

long double Time::DeltaTimePrinting()
{
	return m_deltaTimePrinting.count();
}

void Time::UpdateRendering()
{
	m_deltaTimeRendering = t_clock::now() - m_loopBeginRendering;
	m_loopBeginRendering = t_clock::now();
}

void Time::UpdatePrinting()
{
	m_deltaTimePrinting = t_clock::now() - m_loopBeginPrinting;
	m_loopBeginPrinting = t_clock::now();
}