﻿#include "pch.h"
#include "RayTracing.h"

#define R(c) (((c) >> 16) & 0xff)
#define G(c) (((c) >>  8) & 0xff)
#define B(c) ( (c)        & 0xff)

#define CUBE_THRESHOLDS(a, b, c, d, e)		\
	if      (v < a) return IDX(0,   0);	\
	else if (v < b) return IDX(1,  95);	\
	else if (v < c) return IDX(2, 135);	\
	else if (v < d) return IDX(3, 175);	\
	else if (v < e) return IDX(4, 215);	\
	else            return IDX(5, 255);

#define IDX(i, v) ((((uint32_t)i * 36 + 16) << 24) | ((uint32_t)v << 16))

__device__ static uint32_t cube_index_red(uint8_t v) {
	CUBE_THRESHOLDS(38, 115, 155, 196, 235);
}

#undef IDX
#define IDX(i, v) ((((uint32_t)i * 6) << 24) | ((uint32_t)v << 8))

__device__ static uint32_t cube_index_green(uint8_t v) {
	CUBE_THRESHOLDS(36, 116, 154, 195, 235);
}

#undef IDX
#define IDX(i, v) (((uint32_t)i << 24) | (uint32_t)v)

__device__ static uint32_t cube_index_blue(uint8_t v) {
	CUBE_THRESHOLDS(35, 115, 155, 195, 235);
}

#undef IDX
#undef CUBE_THRESHOLDS


__device__ uint32_t rgb_from_ansi256(uint8_t index) {
	static uint32_t colours[256] = {
		/* The 16 system colours as used by default by xterm.  Taken
		   from XTerm-col.ad distributed with xterm source code. */
		0x000000, 0xcd0000, 0x00cd00, 0xcdcd00,
		0x0000ee, 0xcd00cd, 0x00cdcd, 0xe5e5e5,
		0x7f7f7f, 0xff0000, 0x00ff00, 0xffff00,
		0x5c5cff, 0xff00ff, 0x00ffff, 0xffffff,

		/* 6×6×6 cube.  One each axis, the six indices map to [0, 95,
		   135, 175, 215, 255] RGB component values. */
		0x000000, 0x00005f, 0x000087, 0x0000af,
		0x0000d7, 0x0000ff, 0x005f00, 0x005f5f,
		0x005f87, 0x005faf, 0x005fd7, 0x005fff,
		0x008700, 0x00875f, 0x008787, 0x0087af,
		0x0087d7, 0x0087ff, 0x00af00, 0x00af5f,
		0x00af87, 0x00afaf, 0x00afd7, 0x00afff,
		0x00d700, 0x00d75f, 0x00d787, 0x00d7af,
		0x00d7d7, 0x00d7ff, 0x00ff00, 0x00ff5f,
		0x00ff87, 0x00ffaf, 0x00ffd7, 0x00ffff,
		0x5f0000, 0x5f005f, 0x5f0087, 0x5f00af,
		0x5f00d7, 0x5f00ff, 0x5f5f00, 0x5f5f5f,
		0x5f5f87, 0x5f5faf, 0x5f5fd7, 0x5f5fff,
		0x5f8700, 0x5f875f, 0x5f8787, 0x5f87af,
		0x5f87d7, 0x5f87ff, 0x5faf00, 0x5faf5f,
		0x5faf87, 0x5fafaf, 0x5fafd7, 0x5fafff,
		0x5fd700, 0x5fd75f, 0x5fd787, 0x5fd7af,
		0x5fd7d7, 0x5fd7ff, 0x5fff00, 0x5fff5f,
		0x5fff87, 0x5fffaf, 0x5fffd7, 0x5fffff,
		0x870000, 0x87005f, 0x870087, 0x8700af,
		0x8700d7, 0x8700ff, 0x875f00, 0x875f5f,
		0x875f87, 0x875faf, 0x875fd7, 0x875fff,
		0x878700, 0x87875f, 0x878787, 0x8787af,
		0x8787d7, 0x8787ff, 0x87af00, 0x87af5f,
		0x87af87, 0x87afaf, 0x87afd7, 0x87afff,
		0x87d700, 0x87d75f, 0x87d787, 0x87d7af,
		0x87d7d7, 0x87d7ff, 0x87ff00, 0x87ff5f,
		0x87ff87, 0x87ffaf, 0x87ffd7, 0x87ffff,
		0xaf0000, 0xaf005f, 0xaf0087, 0xaf00af,
		0xaf00d7, 0xaf00ff, 0xaf5f00, 0xaf5f5f,
		0xaf5f87, 0xaf5faf, 0xaf5fd7, 0xaf5fff,
		0xaf8700, 0xaf875f, 0xaf8787, 0xaf87af,
		0xaf87d7, 0xaf87ff, 0xafaf00, 0xafaf5f,
		0xafaf87, 0xafafaf, 0xafafd7, 0xafafff,
		0xafd700, 0xafd75f, 0xafd787, 0xafd7af,
		0xafd7d7, 0xafd7ff, 0xafff00, 0xafff5f,
		0xafff87, 0xafffaf, 0xafffd7, 0xafffff,
		0xd70000, 0xd7005f, 0xd70087, 0xd700af,
		0xd700d7, 0xd700ff, 0xd75f00, 0xd75f5f,
		0xd75f87, 0xd75faf, 0xd75fd7, 0xd75fff,
		0xd78700, 0xd7875f, 0xd78787, 0xd787af,
		0xd787d7, 0xd787ff, 0xd7af00, 0xd7af5f,
		0xd7af87, 0xd7afaf, 0xd7afd7, 0xd7afff,
		0xd7d700, 0xd7d75f, 0xd7d787, 0xd7d7af,
		0xd7d7d7, 0xd7d7ff, 0xd7ff00, 0xd7ff5f,
		0xd7ff87, 0xd7ffaf, 0xd7ffd7, 0xd7ffff,
		0xff0000, 0xff005f, 0xff0087, 0xff00af,
		0xff00d7, 0xff00ff, 0xff5f00, 0xff5f5f,
		0xff5f87, 0xff5faf, 0xff5fd7, 0xff5fff,
		0xff8700, 0xff875f, 0xff8787, 0xff87af,
		0xff87d7, 0xff87ff, 0xffaf00, 0xffaf5f,
		0xffaf87, 0xffafaf, 0xffafd7, 0xffafff,
		0xffd700, 0xffd75f, 0xffd787, 0xffd7af,
		0xffd7d7, 0xffd7ff, 0xffff00, 0xffff5f,
		0xffff87, 0xffffaf, 0xffffd7, 0xffffff,

		/* Greyscale ramp.  This is calculated as (index - 232) * 10 + 8
		   repeated for each RGB component. */
		0x080808, 0x121212, 0x1c1c1c, 0x262626,
		0x303030, 0x3a3a3a, 0x444444, 0x4e4e4e,
		0x585858, 0x626262, 0x6c6c6c, 0x767676,
		0x808080, 0x8a8a8a, 0x949494, 0x9e9e9e,
		0xa8a8a8, 0xb2b2b2, 0xbcbcbc, 0xc6c6c6,
		0xd0d0d0, 0xdadada, 0xe4e4e4, 0xeeeeee,
	};

	return colours[index];
}

__device__ static uint32_t distance(uint32_t x, uint32_t y) {
	int32_t r_sum = R(x) + R(y);
	int32_t r = (int32_t)R(x) - (int32_t)R(y);
	int32_t g = (int32_t)G(x) - (int32_t)G(y);
	int32_t b = (int32_t)B(x) - (int32_t)B(y);
	return (1024 + r_sum) * r * r + 2048 * g * g + (1534 - r_sum) * b * b;
}

__device__ static uint8_t luminance(uint32_t rgb) {
	
	const uint32_t v = (UINT32_C(3567664) * R(rgb) +
		UINT32_C(11998547) * G(rgb) +
		UINT32_C(1211005) * B(rgb));
	/* Round to nearest rather than truncating when dividing. */
	return (v + (UINT32_C(1) << 23)) >> 24;
}

__device__ uint8_t ansi256_from_rgb(uint32_t rgb) {
	
	static const uint8_t ansi256_from_grey[256] = {
		 16,  16,  16,  16,  16, 232, 232, 232,
		232, 232, 232, 232, 232, 232, 233, 233,
		233, 233, 233, 233, 233, 233, 233, 233,
		234, 234, 234, 234, 234, 234, 234, 234,
		234, 234, 235, 235, 235, 235, 235, 235,
		235, 235, 235, 235, 236, 236, 236, 236,
		236, 236, 236, 236, 236, 236, 237, 237,
		237, 237, 237, 237, 237, 237, 237, 237,
		238, 238, 238, 238, 238, 238, 238, 238,
		238, 238, 239, 239, 239, 239, 239, 239,
		239, 239, 239, 239, 240, 240, 240, 240,
		240, 240, 240, 240,  59,  59,  59,  59,
		 59, 241, 241, 241, 241, 241, 241, 241,
		242, 242, 242, 242, 242, 242, 242, 242,
		242, 242, 243, 243, 243, 243, 243, 243,
		243, 243, 243, 244, 244, 244, 244, 244,
		244, 244, 244, 244, 102, 102, 102, 102,
		102, 245, 245, 245, 245, 245, 245, 246,
		246, 246, 246, 246, 246, 246, 246, 246,
		246, 247, 247, 247, 247, 247, 247, 247,
		247, 247, 247, 248, 248, 248, 248, 248,
		248, 248, 248, 248, 145, 145, 145, 145,
		145, 249, 249, 249, 249, 249, 249, 250,
		250, 250, 250, 250, 250, 250, 250, 250,
		250, 251, 251, 251, 251, 251, 251, 251,
		251, 251, 251, 252, 252, 252, 252, 252,
		252, 252, 252, 252, 188, 188, 188, 188,
		188, 253, 253, 253, 253, 253, 253, 254,
		254, 254, 254, 254, 254, 254, 254, 254,
		254, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 231,
		231, 231, 231, 231, 231, 231, 231, 231,
	};

	/* First of, if it’s shade of grey, we know exactly the best colour that
	   approximates it. */
	if (R(rgb) == G(rgb) && G(rgb) == B(rgb)) {
		return ansi256_from_grey[rgb & 0xff];
	}

	uint8_t grey_index = ansi256_from_grey[luminance(rgb)];
	uint32_t grey_distance = distance(rgb, rgb_from_ansi256(grey_index));
	uint32_t cube = cube_index_red(R(rgb)) + cube_index_green(G(rgb)) +
		cube_index_blue(B(rgb));
	return distance(rgb, cube) < grey_distance ? cube >> 24 : grey_index;
}

__global__ void UpdateObjects(
	Object3D** objects,
	unsigned int count,
	double dt
)
{
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count)
	{
		return;
	}
	//Do culling, oct-tree, occlusion
	//When this is implemented we have to make it its own kernel since all objects oct-tree node has to be updated before physics update starts.

	//Do physics against objects in oct-tree nodes next to this object's node
	Object3D* object = objects[index];
	switch (object->GetType())
	{
	case SphereType:
	{
		((Sphere*)object)->Update(dt);
		break;
	}
	case PlaneType:
	{
		((Plane*)object)->Update(dt);
		break;
	}
	default:
	{
		break;
	}
	}
	return;
}

__global__ void Culling(

)
{

}

__global__ void RT(
	Object3D** objects,
	unsigned int count,
	size_t x,
	size_t y,
	float element1,
	float element2,
	RayTracingParameters* params,
	char* resultArray
)
{
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column >= x || row >= y)
	{
		return;
	}
	if (column == (x - 1))
	{
		resultArray[row * (x * 12) + column * 12] = '\n';
		return;
	}
	//Convert pixel coordinates to (clip space? screen space?)
	float convertedY = ((float)y - row * 2) / y;
	float convertedX = (2 * column - (float)x) / x;

	//Calculate the ray.
	Vector4 pixelVSpace = Vector4(convertedX * element1, convertedY * element2, 1.0f, 0.0f);
	Vector4 tempDirectionWSpace = params->inverseVMatrix.Mult(pixelVSpace);
	Vector3 directionWSpace = Vector3(tempDirectionWSpace.x, tempDirectionWSpace.y, tempDirectionWSpace.z).Normalize();
	
	//Used during intersection tests with spheres.
	float a = Dot(directionWSpace, directionWSpace);
	float fourA = 4.0f * a;
	float divTwoA = 1.0f / (2.0f * a);

	char data = ' ';
	float closest = 99999999.f;
	float shadingValue = 0.0f;
	Vector3 bestColor = Vector3();

	//Localizing variables.
	Vector3 cameraPos = params->camPos;
	size_t localCount = count;
	
	//Ray trace against every object.
	for (size_t i = 0; i < localCount; i++)
	{
		//Localize the current object.
		Object3D localObject = *objects[i];
		//Here we need to check if the object is culled, if it is we continue on the next object.
		

		ObjectType type = localObject.GetType();
		//Ray-Sphere intersection test.
		if (type == SphereType)
		{
			Sphere localSphere = *(Sphere*)(objects[i]);
			Vector3 objectToCam = cameraPos - localSphere.GetPos();
			float radius = localSphere.GetRadius();

			float b = 2.0f * Dot(directionWSpace, objectToCam);
			float c = Dot(objectToCam, objectToCam) - (radius * radius);

			float discriminant = b * b - fourA * c;

			//It hit
			if (discriminant >= 0.0f)
			{
				float sqrtDiscriminant = sqrt(discriminant);
				float minusB = -b;
				float t1 = (minusB + sqrtDiscriminant) * divTwoA;
				float t2 = (minusB - sqrtDiscriminant) * divTwoA;

				float closerPoint = 0.0f;
				if (t1 <= t2)
				{
					closerPoint = t1;
				}
				else
				{
					closerPoint = t2;
				}

				if (closerPoint < closest && closerPoint > 0.0f)
				{
					closest = closerPoint;

					Vector3 normalSphere = (Vector3(cameraPos.x + directionWSpace.x * closerPoint, cameraPos.y + directionWSpace.y * closerPoint, cameraPos.z + directionWSpace.z * closerPoint) - localSphere.GetPos()).Normalize();
					shadingValue = abs(Dot(normalSphere, Vector3() - directionWSpace));
					bestColor = localSphere.GetColor();
				}
			}
		}
		else if (type == PlaneType)
		{
			Plane localPlane = *((Plane*)(objects[i]));
			//Check if they are paralell, if not it hit.
			Vector3 planeNormal = localPlane.GetNormal();
			float dotLineAndPlaneNormal = Dot(directionWSpace, planeNormal);
			if (dotLineAndPlaneNormal != 0.0f)
			{
				float t1 = Dot((localPlane.GetPos() - cameraPos), planeNormal) / dotLineAndPlaneNormal;

				if (t1 > 0.0f)
				{

					if (t1 < closest)
					{
						Vector3 p = cameraPos + (directionWSpace * t1);
						if (p.x > -7.0f && p.x < 7.0f && p.z > 12.0f && p.z < 35.0f) //Just arbitrary restictions. Put these into plane instead.
						{
							shadingValue = abs(Dot(planeNormal, Vector3() - directionWSpace)); //Remove abs here and comment in the if-statement to get backface culling for planes.
							//if (shadingValue > 0.0f) {
							closest = t1;
							//}
							bestColor = localPlane.GetColor();
						}
					}
				}
			}
			
		}
	}

	//Dont open this. Here be dragons.
	{
		//I warned u
	//$ @B% 8&W M#* oah kbd pqw mZO 0QL CJU YXz cvu nxr jft /| ()1 { } [ ]?- _+~ < >i!lI ; : ,"^`.
		float t = 0.01492537f;
		if (shadingValue < 0.00001f)
		{
			data = ' ';
		}
		else if (shadingValue < t * 1)
		{
			data = '.';
		}
		else if (shadingValue < t * 2)
		{
			data = '`';
		}
		else if (shadingValue < t * 3)
		{
			data = '^';
		}
		else if (shadingValue < t * 4)
		{
			data = '"';
		}
		else if (shadingValue < t * 5)
		{
			data = ',';
		}
		else if (shadingValue < t * 6)
		{
			data = ':';
		}
		else if (shadingValue < t * 7)
		{
			data = ';';
		}
		else if (shadingValue < t * 8)
		{
			data = 'I';
		}
		else if (shadingValue < t * 9)
		{
			data = 'l';
		}
		else if (shadingValue < t * 10)
		{
			data = '!';
		}
		else if (shadingValue < t * 11)
		{
			data = 'i';
		}
		else if (shadingValue < t * 12)
		{
			data = '>';
		}
		else if (shadingValue < t * 13)
		{
			data = '<';
		}
		else if (shadingValue < t * 14)
		{
			data = '~';
		}
		else if (shadingValue < t * 15)
		{
			data = '+';
		}
		else if (shadingValue < t * 16)
		{
			data = '_';
		}
		else if (shadingValue < t * 17)
		{
			data = '-';
		}
		else if (shadingValue < t * 18)
		{
			data = '?';
		}
		else if (shadingValue < t * 19)
		{
			data = '*';
		}
		else if (shadingValue < t * 20)
		{
			data = ']';
		}
		else if (shadingValue < t * 21)
		{
			data = '[';
		}
		else if (shadingValue < t * 22)
		{
			data = '}';
		}
		else if (shadingValue < t * 23)
		{
			data = '{';
		}
		else if (shadingValue < t * 24)
		{
			data = '1';
		}
		else if (shadingValue < t * 25)
		{
			data = ')';
		}
		else if (shadingValue < t * 26)
		{
			data = '(';
		}
		else if (shadingValue < t * 27)
		{
			data = '|';
		}
		else if (shadingValue < t * 28)
		{
			data = '/';
		}
		else if (shadingValue < t * 29)
		{
			data = 't';
		}
		else if (shadingValue < t * 30)
		{
			data = 'f';
		}
		else if (shadingValue < t * 31)
		{
			data = 'j';
		}
		else if (shadingValue < t * 32)
		{
			data = 'r';
		}
		else if (shadingValue < t * 33)
		{
			data = 'x';
		}
		else if (shadingValue < t * 34)
		{
			data = 'n';
		}
		else if (shadingValue < t * 35)
		{
			data = 'u';
		}
		else if (shadingValue < t * 36)
		{
			data = 'v';
		}
		else if (shadingValue < t * 37)
		{
			data = 'c';
		}
		else if (shadingValue < t * 38)
		{
			data = 'z';
		}
		else if (shadingValue < t * 39)
		{
			data = 'm';
		}
		else if (shadingValue < t * 40)
		{
			data = 'w';
		}
		else if (shadingValue < t * 41)
		{
			data = 'X';
		}
		else if (shadingValue < t * 42)
		{
			data = 'Y';
		}
		else if (shadingValue < t * 43)
		{
			data = 'U';
		}
		else if (shadingValue < t * 44)
		{
			data = 'J';
		}
		else if (shadingValue < t * 45)
		{
			data = 'C';
		}
		else if (shadingValue < t * 46)
		{
			data = 'L';
		}
		else if (shadingValue < t * 47)
		{
			data = 'q';
		}
		else if (shadingValue < t * 48)
		{
			data = 'p';
		}
		else if (shadingValue < t * 49)
		{
			data = 'd';
		}
		else if (shadingValue < t * 50)
		{
			data = 'b';
		}
		else if (shadingValue < t * 51)
		{
			data = 'k';
		}
		else if (shadingValue < t * 52)
		{
			data = 'h';
		}
		else if (shadingValue < t * 53)
		{
			data = 'a';
		}
		else if (shadingValue < t * 54)
		{
			data = 'o';
		}
		else if (shadingValue < t * 55)
		{
			data = '#';
		}
		else if (shadingValue < t * 56)
		{
			data = '%';
		}
		else if (shadingValue < t * 57)
		{
			data = 'Z';
		}
		else if (shadingValue < t * 58)
		{
			data = 'O';
		}
		else if (shadingValue < t * 59)
		{
			data = '8';
		}
		else if (shadingValue < t * 60)
		{
			data = 'B';
		}
		else if (shadingValue < t * 61)
		{
			data = '$';
		}
		else if (shadingValue < t * 62)
		{
			data = '0';
		}
		else if (shadingValue < t * 63)
		{
			data = 'Q';
		}
		else if (shadingValue < t * 64)
		{
			data = 'M';
		}
		else if (shadingValue < t * 65)
		{
			data = '&';
		}
		else if (shadingValue < t * 66)
		{
			data = 'W';
		}
		else
		{
			data = '@';
		}
	}
	
	//Allow different printing modes.
	if (true) //Pixels only.
	{
		if (data != ' ')
		{
			bestColor.x *= shadingValue;
			bestColor.y *= shadingValue;
			bestColor.z *= shadingValue;
			//int index = (bestColor.x * 6 / 256) * 36 + (bestColor.y * 6 / 256) * 6 + (bestColor.z * 6 / 256);
			//int index = (int)(bestColor.x * 7 / 255) << 5 + (int)(bestColor.y * 7 / 255) << 2 + (int)(bestColor.z * 3 / 255);
			uint8_t index = ansi256_from_rgb(((uint8_t)bestColor.x << 16) + ((uint8_t)bestColor.y << 8) + (uint8_t)bestColor.z);
			uint8_t tens = index % 100;
			uint8_t singles = tens % 10;
			char first = '\0';
			char second = '\0';
			char third = '\0';

			if (index >= 100)
			{
				index = (uint8_t)((index - tens) / 100);
				first = index + '0';
			}
			if (tens >= 10 || index >= 100)
			{
				tens = (uint8_t)((tens - singles) / 10);
				second = tens + '0';
			}
			third = singles + '0';
			char finalData[12] = {
			'\x1b', '[',			//Escape character
			'3', '8', ';',			//Keycode for background
			'5', ';',				//Keycode for background
			first, second, third,//charIndex[0], charIndex[1], charIndex[2],			//Index - convert RGB to 8bit = encodedData = (Math.floor((red / 32)) << 5) + (Math.floor((green / 32)) << 2) + Math.floor((blue / 64));
			'm', data				//Character data.
			};
			memcpy(resultArray + (row * (x * 12) + column * 12), finalData, sizeof(char) * 12);
		}
		else
		{
			char finalData[12] = {
			'\x1b', '[',			//Escape character
			'4', '8', ';',			//Keycode for background
			'5', ';',				//Keycode for background
			'0', '\0', '\0',			//Index - convert RGB to 8bit = encodedData = (Math.floor((red / 32)) << 5) + (Math.floor((green / 32)) << 2) + Math.floor((blue / 64));
			'm', ' '				//Character data.
			};
			memcpy(resultArray + (row * (x * 12) + column * 12), finalData, sizeof(char) * 12);
		}
	}
	else //Ascii signs with color.
	{
		char finalData[12] = {
			'\x1b', '[',			//Escape character
			'3', '8', ';',			//Keycode for foreground
			'5', ';',				//Keycode for foreground
			'2', '5', '5',			//Index 
			'm', data				//Character data.
		};
		memcpy(resultArray + (row * (x * 12) + column * 12), finalData, sizeof(char) * 12);
	}
	//memcpy(resultArray + (row * (x * 37) + column * 37), finalData, sizeof(char) * 37);
	//resultArray[row * (x * 37) + column * 37] = data;
	return;
	
}

void RayTracingWrapper(size_t x, size_t y, float element1, float element2, DeviceObjectArray<Object3D*> deviceObjects, RayTracingParameters* deviceParams, char* deviceResultArray, std::mutex* backBufferMutex, double dt)
{
	//Update the objects. 1 thread per object.
	unsigned int threadsPerBlock = deviceObjects.count;
	unsigned int numberOfBlocks = 1;
	if (deviceObjects.count > 1024)
	{
		numberOfBlocks = std::ceil(deviceObjects.count / 1024.0);
	}
	dim3 gridDims(numberOfBlocks, 1, 1);
	dim3 blockDims(threadsPerBlock, 1, 1);
	
	//If it is the first rendering loop we need to construct the octtree, so that we can access it in the physicsupdate. But otherwise it only has to be done after the physics update.
	// 
	//Physics update of the objects.
	UpdateObjects<<<gridDims, blockDims>>>(
		deviceObjects.using1st ? deviceObjects.m_deviceArray1 : deviceObjects.m_deviceArray2,
		deviceObjects.count,
		dt
	);
	/*
	Culling<<<gridDims, blockDims>>>(
		deviceObjects.using1st ? deviceObjects.m_deviceArray1 : deviceObjects.m_deviceArray2,

	);
	*/
	//Do the raytracing. Calculate x and y dimensions in blocks depending on screensize.
	//1 thread per pixel.
	gridDims.x = std::ceil((float)(x + 1) / 16.0);
	gridDims.y = std::ceil((float)y / 16.0);
	blockDims.x = 16;
	blockDims.y = 16;
	
	RT<<<gridDims, blockDims>>>(
		deviceObjects.using1st ? deviceObjects.m_deviceArray1 : deviceObjects.m_deviceArray2,
		deviceObjects.count,
		x,
		y,
		element1,
		element2,
		deviceParams,
		deviceResultArray
	);
	//Make sure all the threads are done.
	//Then we lock the mutex and copy the results from the GPU to the backbuffer.
	//Then we signal to the print thread that the backbuffer is ready. 
	gpuErrchk(cudaDeviceSynchronize());
	backBufferMutex->lock();
	PrintMachine* printMachine = PrintMachine::GetInstance();
	char* backBuffer = printMachine->GetBackBuffer();
	cudaMemcpy(backBuffer, deviceResultArray, (12 * printMachine->GetWidth() * printMachine->GetHeight()) + printMachine->GetHeight(), cudaMemcpyDeviceToHost);
	memset(printMachine->GetBackBufferSwap(), 1, sizeof(int));
	backBufferMutex->unlock();
	return;
}