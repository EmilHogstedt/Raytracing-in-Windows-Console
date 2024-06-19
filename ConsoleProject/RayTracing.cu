#include "pch.h"
#include "RayTracing.h"

//Move all this to a different file.
//Not allowed to change this code without making it GNU LGPL
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

__constant__ uint32_t colours[256] = {
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

__device__ uint32_t rgb_from_ansi256(uint8_t index) {
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
	
	//The first option out of these 2 is 5 times faster but less accurate.
	return (v + (UINT32_C(1) << 23)) >> 24;
	/*
	return sqrtf((float)R(rgb) * (float)R(rgb) * 0.2126729f +
		(float)G(rgb) * (float)G(rgb) * 0.7151521f +
		(float)B(rgb) * (float)B(rgb) * 0.0721750);
	*/
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
	case ObjectType::SphereType:
	{
		((Sphere*)object)->Update(dt);
		break;
	}
	case ObjectType::PlaneType:
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
	float camFarDist,
	RayTracingParameters* params,
	char* resultArray,
	PrintMachine::PrintMode mode
)
{
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t column = blockIdx.x * blockDim.x + threadIdx.x;
	size_t size = 12; //if the mode is pixel or ascii
	if (mode != PrintMachine::ASCII && mode != PrintMachine::PIXEL)
	{
		size = 20; //if the mode is rgb we need more characters.
	}
	
	if (column >= x || row >= y)
	{
		return;
	}
	
	if (column == (x - 1))
	{
		resultArray[row * (x * size) + column * size] = '\n';
		return;
	}
	//Convert pixel coordinates to (clip space? screen space?)
	float convertedY = ((float)y - row * 2) / y;
	float convertedX = (2 * column - (float)x) / x;

	//Calculate the ray.
	MyMath::Vector4 pixelVSpace = MyMath::Vector4(convertedX * element1, convertedY * element2, 1.0f, 0.0f);
	MyMath::Vector3 directionWSpace = params->inverseVMatrix.Mult(pixelVSpace).xyz().Normalize();
	
	//Used during intersection tests with spheres.
	float a = Dot(directionWSpace, directionWSpace);
	float fourA = 4.0f * a;
	float divTwoA = 1.0f / (2.0f * a);

	char data = ' ';
	float closest = 99999999.f;
	float shadingValue = 0.0f;
	MyMath::Vector3 bestColor;
	MyMath::Vector3 bestNormal;

	//Localizing variables.
	MyMath::Vector3 cameraPos = params->camPos;
	size_t localCount = count;
	
	//Ray trace against every object.
	for (size_t i = 0; i < localCount; i++)
	{
		//Localize the current object.
		Object3D localObject = *objects[i];
		//Here we need to check if the object is culled, if it is we continue on the next object.
		

		ObjectType type = localObject.GetType();
		//Ray-Sphere intersection test.
		if (type == ObjectType::SphereType)
		{
			Sphere localSphere = *(Sphere*)(objects[i]);
			MyMath::Vector3 spherePos = localSphere.GetPos();

			MyMath::Vector3 objectToCam = cameraPos - spherePos;
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

				//Remove second condition to enable "backface" culling for spheres. IE; not hit when inside them.
				if (t1 > t2 && t2 >= 0.0f)
				{
					t1 = t2;
				}

				if (t1 < closest && t1 > 0.0f)
				{
					closest = t1;
					MyMath::Vector3 normalSphere = (cameraPos + directionWSpace * closest - spherePos).Normalize();
					bestNormal = normalSphere;

					//The vector 3 here is just to make the spheres not "follow" the player.
					shadingValue = Dot(normalSphere, MyMath::Vector3(1.0f, 0.0f, 0.0f));
					bestColor = localSphere.GetColor();
				}
			}
		}
		else if (type == ObjectType::PlaneType)
		{
			Plane localPlane = *((Plane*)(objects[i]));
			MyMath::Vector3 planeNormal = localPlane.GetNormal();
			//Check if they are paralell, if not it hit.
			float dotLineAndPlaneNormal = Dot(directionWSpace, planeNormal);
			if (dotLineAndPlaneNormal != 0.0f)
			{
				float t1 = Dot((localPlane.GetPos() - cameraPos), planeNormal) / dotLineAndPlaneNormal;

				if (t1 > 0.0f)
				{
					if (t1 < closest)
					{
						MyMath::Vector3 p = cameraPos + (directionWSpace * t1);
						if (p.x > -7.0f && p.x < 7.0f && p.z > 12.0f && p.z < 35.0f) //Just arbitrary restictions. Put these into plane instead.
						{
							shadingValue = Dot(planeNormal, MyMath::Vector3(1.0f, 0.0f, 0.0f));

							//Comment in this if statement to get "backface" culling for planes.
							//if (shadingValue > 0.0f) {
							closest = t1;
							//}
							bestColor = localPlane.GetColor();
							bestNormal = planeNormal;
							if (dotLineAndPlaneNormal > 0.0f)
							{
								bestNormal *= -1;
							}
						}
					}
				}
			}
			
		}
	}

	//Dont open this. Here be dragons. Print characters for ASCII mode.
	{
		//I warned u
	//$ @B% 8&W M#* oah kbd pqw mZO 0QL CJU YXz cvu nxr jft /| ()1 { } [ ]?- _+~ < >i!lI ; : ,"^`.
		float t = 0.01492537f;
		//If we miss or its outside the frustum we dont print anything.
		if (closest > camFarDist)
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
	
	//Now we need to take the raytraced information and output it to our result array of chars.
	//If the mode is not RGB we need to convert the colors to 8bit.
	if (mode == PrintMachine::PIXEL || mode == PrintMachine::ASCII)
	{
		//If the pixel hit something during ray tracing.
		if (data != ' ')
		{
			float ambient = 0.01492537f * 19;
			if (shadingValue < ambient)
			{
				shadingValue = ambient;
			}
			//Apply shading.
			bestColor *= shadingValue;

			//Convert the 24bit RGB color to ANSI 8 bit color.
			uint8_t index = ansi256_from_rgb(((uint8_t)bestColor.x << 16) + ((uint8_t)bestColor.y << 8) + (uint8_t)bestColor.z);
			uint8_t originalIndex = index;
			//Now we need to convert this number (0-255) to 3 chars.
			uint8_t tens = index % 100;
			uint8_t singles = tens % 10;
			char first = '\0';
			char second = '\0';
			char third = '\0';

			if (index >= 100)
			{
				index = (uint8_t)((index - tens) * 0.01f);
				first = index + '0';
			}
			if (tens >= 10 || originalIndex >= 100)
			{
				tens = (uint8_t)((tens - singles) * 0.1f);
				second = tens + '0';
			}
			third = singles + '0';

			//If in ASCII mode we change foreground color and also print the value in data.
			if (mode == PrintMachine::ASCII)
			{
				char finalData[12] = {
					'\x1b', '[',			//Escape character
					'3', '8', ';',			//Keycode for foreground
					'5', ';',				//Keycode for foreground
					first, second, third,	//Index
					'm', data				//Character data.
				};
				memcpy(resultArray + (row * (x * size) + column * size), finalData, sizeof(char) * size);
			}
			//If in PIXEL mode we change background color and do not print the value.
			else //If in pixel mode we only print the color.
			{
				char finalData[12] = {
					'\x1b', '[',			//Escape character
					'4', '8', ';',			//Keycode for background
					'5', ';',				//Keycode for background
					first, second, third,	//Index
					'm', ' '				//Character data.
				};
				memcpy(resultArray + (row * (x * size) + column * size), finalData, sizeof(char) * size);
			}
		}
		//If it is an empty space we can not use a background color.
		else
		{
			char finalData[12] = {
				'\x1b', '[',			//Escape character
				'4', '8', ';',			//Keycode for background
				'5', ';',				//Keycode for background
				'\0', '1', '6',			//Index
				'm', ' '				//Character data.
			};
			memcpy(resultArray + (row * (x * size) + column * size), finalData, sizeof(char) * size);
		}
	}
	//If the mode is in any of the RGB modes we simply use the rgb values gathered.
	else
	{
		//If the pixel hit something during ray tracing.
		if (data != ' ')
		{
			//Increase the right-hand value to increase the ambient light.
			float ambient = 0.01492537f * 7;
			if (shadingValue < ambient)
			{
				shadingValue = ambient;
			}
			//Apply shading.
			bestColor *= shadingValue;

			//Needed to print the rgb values to final data.
			char firstR = '\0';
			char secondR = '\0';
			char thirdR = '\0';

			char firstG = '\0';
			char secondG = '\0';
			char thirdG = '\0';

			char firstB = '\0';
			char secondB = '\0';
			char thirdB = '\0';

			//R
			uint8_t originalIndex;
			uint8_t index;
			if (mode == PrintMachine::RGB_NORMALS)
			{
				originalIndex = (uint8_t)(bestNormal.x * 255);
				index = (uint8_t)(bestNormal.x * 255);
			}
			else
			{
				originalIndex = (uint8_t)bestColor.x;
				index = (uint8_t)bestColor.x;
			}
			
			uint8_t tens = index % 100;
			uint8_t singles = tens % 10;

			if (index >= 100)
			{
				index = (uint8_t)((index - tens) * 0.01f);
				firstR = index + '0';
			}
			if (tens >= 10 || originalIndex >= 100)
			{
				tens = (uint8_t)((tens - singles) * 0.1f);
				secondR = tens + '0';
			}
			thirdR = singles + '0';

			//G
			if (mode == PrintMachine::RGB_NORMALS)
			{
				originalIndex = (uint8_t)(bestNormal.y * 255);
				index = (uint8_t)(bestNormal.y * 255);
			}
			else
			{
				originalIndex = (uint8_t)bestColor.y;
				index = (uint8_t)bestColor.y;
			}

			tens = index % 100;
			singles = tens % 10;

			if (index >= 100)
			{
				index = (uint8_t)((index - tens) * 0.01f);
				firstG = index + '0';
			}
			if (tens >= 10 || originalIndex >= 100)
			{
				tens = (uint8_t)((tens - singles) * 0.1f);
				secondG = tens + '0';
			}
			thirdG = singles + '0';

			//B
			if (mode == PrintMachine::RGB_NORMALS)
			{
				originalIndex = (uint8_t)(bestNormal.z * 255);
				index = (uint8_t)(bestNormal.z * 255);
			}
			else
			{
				originalIndex = (uint8_t)bestColor.z;
				index = (uint8_t)bestColor.z;
			}

			tens = index % 100;
			singles = tens % 10;

			if (index >= 100)
			{
				index = (uint8_t)((index - tens) * 0.01f);
				firstB = index + '0';
			}
			if (tens >= 10 || originalIndex >= 100)
			{
				tens = (uint8_t)((tens - singles) * 0.1f);
				secondB = tens + '0';
			}
			thirdB = singles + '0';

			//If in ASCII mode we change foreground color and also print the value in data.
			if (mode == PrintMachine::RGB_ASCII)
			{
				char finalData[20] = {
					'\x1b', '[',					//Escape character
					'3', '8', ';',					//Keycode for foreground
					'2', ';',						//Keycode for foreground
					firstR, secondR, thirdR, ';',	//R
					firstG, secondG, thirdG, ';',	//G
					firstB, secondB, thirdB,		//B
					'm', data						//Character data.
				};
				memcpy(resultArray + (row * (x * size) + column * size), finalData, sizeof(char)* size);
			}
			else if (mode == PrintMachine::RGB_PIXEL)
			{
				char finalData[20] = {
					'\x1b', '[',					//Escape character
					'4', '8', ';',					//Keycode for foreground
					'2', ';',						//Keycode for foreground
					firstR, secondR, thirdR, ';',	//R
					firstG, secondG, thirdG, ';',	//G
					firstB, secondB, thirdB,		//B
					'm', ' '						//Character data.
				};
				memcpy(resultArray + (row * (x * size) + column * size), finalData, sizeof(char)* size);
			}
			//Normals.
			else
			{
				char finalData[20] = {
					'\x1b', '[',					//Escape character
					'4', '8', ';',					//Keycode for foreground
					'2', ';',						//Keycode for foreground
					firstR, secondR, thirdR, ';',	//R
					firstG, secondG, thirdG, ';',	//G
					firstB, secondB, thirdB,		//B
					'm', ' '						//Character data.
				};
				memcpy(resultArray + (row * (x * size) + column * size), finalData, sizeof(char) * size);
			}
		}
		//If it is an empty space we can not use a background color. 
		else
		{
			char finalData[20] = {
				'\x1b', '[',			//Escape character
				'4', '8', ';',			//Keycode for background
				'2', ';',				//Keycode for background
				'\0', '\0', '0', ';',	//R
				'\0', '\0', '0', ';',	//G
				'\0', '\0', '0',		//B
				'm', ' '				//Character data.
			};
			memcpy(resultArray + (row * (x * size) + column * size), finalData, sizeof(char)* size);
		}
	}
	
	return;
}

void RayTracer::RayTracingWrapper(size_t x, size_t y, float element1, float element2, float camFarDist, DeviceObjectArray<Object3D*> deviceObjects, RayTracingParameters* deviceParams, char* deviceResultArray, std::mutex* backBufferMutex, double dt)
{
	//Update the objects. 1 thread per object.
	unsigned int threadsPerBlock = deviceObjects.count;
	unsigned int numberOfBlocks = 1;
	if (deviceObjects.count > 1024)
	{
		numberOfBlocks = static_cast<int>(std::ceil(deviceObjects.count / 1024.0));
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

	//Classify the objects into the octtree.
	//Mark objects within the frustum
	/*
	Culling<<<gridDims, blockDims>>>(
		deviceObjects.using1st ? deviceObjects.m_deviceArray1 : deviceObjects.m_deviceArray2,

	);
	*/
	//After we do the culling we check the remaining objects within the octtree and update their closest position to the camera.


	//Do the raytracing. Calculate x and y dimensions in blocks depending on screensize.
	//1 thread per pixel.
	gridDims.x = static_cast<unsigned int>(std::ceil((float)(x + 1) / 16.0));
	gridDims.y = static_cast<unsigned int>(std::ceil((float)y / 16.0));
	blockDims.x = 16u;
	blockDims.y = 16u;
	
	RT << <gridDims, blockDims >> > (
		deviceObjects.using1st ? deviceObjects.m_deviceArray1 : deviceObjects.m_deviceArray2,
		deviceObjects.count,
		x,
		y,
		element1,
		element2,
		camFarDist,
		deviceParams,
		deviceResultArray,
		PrintMachine::GetPrintMode()
	);
	//Make sure all the threads are done.
	//Then we lock the mutex and copy the results from the GPU to the backbuffer.
	//Then we signal to the print thread that the backbuffer is ready. 
	gpuErrchk(cudaDeviceSynchronize());

	//#todo: Make a function to get the original "max size". Why?
	size_t size = PrintMachine::GetMaxSize();

	//memset(m_hostResultArray, '0', size); //Might not be needed. Yep, seems to not be needed.
	//memset(m_minimizedResultArray, '0', size); //Might not be needed. Yep, seems to not be needed.

	gpuErrchk(cudaMemcpy(m_hostResultArray, deviceResultArray, size, cudaMemcpyDeviceToHost));
	size_t newSize = MinimizeResults(size, y);

	PrintMachine::GetBackBufferMutex()->lock();
	//PrintMachine::ResetBackBuffer(); //Might not be needed. Yep, seems to not be needed.
	char* backBuffer = PrintMachine::GetBackBuffer();
	memcpy(backBuffer, m_minimizedResultArray, newSize);
	
	PrintMachine::FlagForBufferSwap();
	PrintMachine::SetPrintSize(newSize);
	PrintMachine::GetBackBufferMutex()->unlock();

	return;
}

size_t RayTracer::MinimizeResults(size_t size, size_t y)
{
	PrintMachine::PrintMode mode = PrintMachine::GetPrintMode();

	size_t newlines = 0;
	size_t addedChars = 0;
	//If its in 8 bit mode.
	if (mode == PrintMachine::ASCII || mode == PrintMachine::PIXEL)
	{
		char latestColor[3] = { 'x', 'x', 'x' };

		for (size_t i = 0; i < size;)
		{
			char current = m_hostResultArray[i];
			if (current == '\x1b')
			{
				//If its not the same color add the whole escape sequence and update latest color.
				if (latestColor[0] != m_hostResultArray[i + 7] || latestColor[1] != m_hostResultArray[i + 8] || latestColor[2] != m_hostResultArray[i + 9])
				{
					memcpy(latestColor, m_hostResultArray + i + 7, 3);

					memcpy(m_minimizedResultArray + addedChars, m_hostResultArray + i, 12);
					addedChars += 12;
				}
				//Only add the data and not the escape sequence.
				else
				{
					//memcpy(m_minimizedResultArray + addedChars, m_hostResultArray + i + 11, 1);
					m_minimizedResultArray[addedChars] = m_hostResultArray[i + 11];
					addedChars += 1;
				}
				i += 12;
			}
			else if (current == '\n')
			{
				newlines++;

				m_minimizedResultArray[addedChars] = '\n';
				addedChars++;
				i++;

				if (newlines == y)
				{
					break;
				}
			}
			//For \0.
			else
			{
				i++;
			}
		}
	}
	//Else it is rgb.
	else
	{
		char latestColor[9] = { 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x' };

		for (size_t i = 0; i < size;)
		{
			char current = m_hostResultArray[i];
			if (current == '\x1b')
			{
				//If its not the same color Add the escape sequence and update latest color.
				if (latestColor[0] != m_hostResultArray[i + 7] || latestColor[1] != m_hostResultArray[i + 8] || latestColor[2] != m_hostResultArray[i + 9] ||
					latestColor[3] != m_hostResultArray[i + 11] || latestColor[4] != m_hostResultArray[i + 12] || latestColor[5] != m_hostResultArray[i + 13] ||
					latestColor[6] != m_hostResultArray[i + 15] || latestColor[7] != m_hostResultArray[i + 16] || latestColor[8] != m_hostResultArray[i + 17])
				{
					memcpy(latestColor, m_hostResultArray + i + 7, 3);
					memcpy(latestColor + 3, m_hostResultArray + i + 11, 3);
					memcpy(latestColor + 6, m_hostResultArray + i + 15, 3);

					memcpy(m_minimizedResultArray + addedChars, m_hostResultArray + i, 20);
					addedChars += 20;
				}
				//Only add the data and not the escape sequence.
				else
				{
					m_minimizedResultArray[addedChars] = m_hostResultArray[i + 19];
					addedChars += 1;
				}
				i += 20;
			}
			else if (current == '\n')
			{
				newlines++;

				m_minimizedResultArray[addedChars] = m_hostResultArray[i];
				addedChars++;
				i++;

				if (newlines == PrintMachine::GetHeight())
				{
					break;
				}
			}
			//For \0.
			else
			{
				i++;
			}
		}
	}
	
	return addedChars;
}