#pragma once

class Time
{
	using t_clock = std::chrono::high_resolution_clock;
	using t_dSec = std::chrono::duration<long double, std::ratio<1, 1>>;
	using t_dNano = std::chrono::duration<long double, std::ratio<1, 1000000000>>;
	using t_moment = std::chrono::time_point<t_clock, t_dNano>;

public:
	Time();
	virtual ~Time() = default;
	long double SinceStart();
	long double DeltaTimeRendering();
	long double DeltaTimePrinting();
	void UpdateRendering();
	void UpdatePrinting();

private:
	t_moment m_start;
	t_moment m_loopBeginRendering;
	t_moment m_loopBeginPrinting;
	t_dSec m_deltaTimeRendering;
	t_dSec m_deltaTimePrinting;
};