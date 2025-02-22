#pragma once

#include "MyMath.h"

enum RenderingMode;
class Object3D;
struct RayTracingParameters;

struct alignas(32) TraceData
{
	MyMath::Vector3 color;
	MyMath::Vector3 normal;
	float distance = 99999999.f;
	float shadingValue;
};

class RayTracing
{
public:
	RayTracing() = delete;
	~RayTracing() = delete;

	static void RayTrace(
		const dim3& gridDims,
		const dim3& blockDims,
		Object3D* DEVICE_MEMORY_PTR const objects,
		const unsigned int count,
		const RayTracingParameters* params,
		char* resultArray,
		const RenderingMode mode);

private:
	
};

__device__
MyMath::Vector3 CalculateInitialDirection(const RayTracingParameters* params);

__device__
char GetASCIICharacter(const float distance, const float farPlane, const float shadingValue);

__device__
void Trace(
	const MyMath::Vector3& direction,
	const MyMath::Vector3& origin,
	const unsigned int count,
	Object3D* DEVICE_MEMORY_PTR const objects,
	TraceData& traceData);

__global__
void RayTrace_ASCII(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingParameters* params,
	char* resultArray);

__global__
void RayTrace_PIXEL(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingParameters* params,
	char* resultArray);

__global__
void RayTrace_RGB_ASCII(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingParameters* params,
	char* resultArray);

__global__
void RayTrace_RGB_PIXEL(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingParameters* params,
	char* resultArray);

__global__
void RayTrace_RGB_NORMALS(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingParameters* params,
	char* resultArray);

__global__
void RayTrace_SDL(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingParameters* params,
	char* resultArray);

#define NUM_ASCII_CHARACTERS 68

__constant__
const char ASCII[NUM_ASCII_CHARACTERS] = {
	' ', '.', '`', '^', '"',
	',', ':', ';', 'I', 'l',
	'!', 'i', '>', '<', '~',
	'+', '_', '-', '?', '*',
	']', '[', '}', '{', '1',
	')', '(', '|', '/', 't',
	'f', 'j', 'r', 'x', 'n',
	'u', 'v', 'c', 'z', 'm',
	'w', 'X', 'Y', 'U', 'J',
	'C', 'L', 'q', 'p', 'd',
	'b', 'k', 'h', 'a', 'o',
	'#', '%', 'Z', 'O', '8',
	'B', '$', '0', 'Q', 'M',
	'&', 'W', '@'
};

__constant__
const float AMBIENT_LIGHT = 0.01492537f * 19;

__constant__
const size_t SIZE_8BIT = 12;

__constant__
const size_t SIZE_RGB = 20;