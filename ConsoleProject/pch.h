#pragma once
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <wchar.h>

#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <limits>

#include <thread>
#include <mutex>
#include <functional>
#include <future>
#include <assert.h>
#include <tchar.h>
#include <memory>

#include <cstdlib>
#include <time.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#if defined(DEBUG) | defined (_DEBUG)
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#else 
#define DBG_NEW new 
#endif