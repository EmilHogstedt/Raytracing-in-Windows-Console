#pragma once
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>

#include <iostream>
#include <vector>
#include <string>

#include <thread>
#include <mutex>

#include <assert.h>
#include <tchar.h>
#include <memory>

#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#if defined(DEBUG) | defined (_DEBUG)
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#else 
#define DBG_NEW new 
#endif