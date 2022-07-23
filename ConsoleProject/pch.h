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
#include <sstream>
#include <limits>

#include <thread>
#include <mutex>
#include <functional>
#include <future>
#include <assert.h>
#include <tchar.h>
#include <memory>

#include <cstdlib>
#include <cstdarg>

#include <time.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#if defined(DEBUG) | defined (_DEBUG)
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#else 
#define DBG_NEW new 
#endif


#define FIVE_MEGABYTES 5000000
#define HUNDRED_MEGABYTES 100000000

#define GRID_DIMENSIONS 21 //NEEDS TO BE UNEVEN

#include <cuda_runtime.h>
#include <cuda.h>
#include <string.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}