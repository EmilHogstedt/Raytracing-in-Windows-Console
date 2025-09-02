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
#include <chrono>

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

#define DEVICE_MEMORY_PTR *

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

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define blockIdxX NULL
#define blockIdxY NULL
#define blockDimX NULL
#define blockDimY NULL
#define threadIdxX NULL
#define threadIdxY NULL
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define blockIdxX blockIdx.x
#define blockIdxY blockIdx.y
#define blockDimX blockDim.x
#define blockDimY blockDim.y
#define threadIdxX threadIdx.x
#define threadIdxY threadIdx.y
#endif

