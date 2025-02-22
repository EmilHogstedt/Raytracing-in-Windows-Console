#include "pch.h"
#include "RayTracing.h"

#include "ANSIRGB.h"
#include "MyMath.h"
#include "Sphere.h"
#include "Plane.h"

char GetASCIICharacter(const float distance, const float farPlane, const float shadingValue)
{
	//If we miss or its outside the frustum we dont print anything.
	if (distance > farPlane)
	{
		return ASCII[0];
	}
	else
	{
		const int dataIndex = MyMath::Clamp((int)ceil(shadingValue * (NUM_ASCII_CHARACTERS - 1)), 1, NUM_ASCII_CHARACTERS); //Clamp with 1 as min, so that the empty space does not get used.
		return ASCII[dataIndex];
	}
}

void Trace(
	const MyMath::Vector3& direction,
	const MyMath::Vector3& origin,
	const unsigned int count,
	Object3D* DEVICE_MEMORY_PTR const objects,
	TraceData& traceData)
{
	//Used during intersection tests with spheres.
	const float a = MyMath::Dot(direction, direction);
	const float fourA = 4.0f * a;
	const float divTwoA = 1.0f / (2.0f * a);

	float closest = 99999999.f;
	float shadingValue = 0.0f;
	MyMath::Vector3 bestColor;
	MyMath::Vector3 bestNormal;

	//Ray trace against every object.
	for (size_t i = 0; i < count; i++)
	{
		//#todo: Here we need to check if the object is culled, if it is we continue on the next object.

		const ObjectType type = objects[i]->GetType();

		//Ray-Sphere intersection test.
		if (type == ObjectType::SphereType)
		{
			const Sphere* sphere = (Sphere*)objects[i];
			const MyMath::Vector3 spherePos = sphere->GetPos();

			const MyMath::Vector3 objectToCam = origin - spherePos;
			const float radius = sphere->GetRadius();

			const float b = 2.0f * Dot(direction, objectToCam);
			const float c = Dot(objectToCam, objectToCam) - (radius * radius);

			const float discriminant = b * b - fourA * c;

			//It hit
			if (discriminant >= 0.0f)
			{
				const float sqrtDiscriminant = sqrt(discriminant);
				const float minusB = -b;
				float t1 = (minusB + sqrtDiscriminant) * divTwoA;
				const float t2 = (minusB - sqrtDiscriminant) * divTwoA;

				//Remove second condition to enable "backface" culling for spheres. IE; not hit when inside them.
				if (t1 > t2 && t2 >= 0.0f)
				{
					t1 = t2;
				}

				if (t1 < closest && t1 > 0.0f)
				{
					closest = t1;
					const MyMath::Vector3 normalSphere = (origin + direction * closest - spherePos).Normalize_GPU();
					bestNormal = normalSphere;

					//1, 0, 0 is just temporary light direction.
					//#todo: INTRODUCE REAL LIGHTS!
					shadingValue = Dot(normalSphere, MyMath::Vector3(1.0f, 0.0f, 0.0f));
					bestColor = sphere->GetColor();
				}
			}
		}
		else if (type == ObjectType::PlaneType)
		{
			const Plane* plane = (Plane*)objects[i];
			const MyMath::Vector3 planeNormal = plane->GetNormal();
			const MyMath::Vector3 planePos = plane->GetPos();

			const float dotLineAndPlaneNormal = Dot(direction, planeNormal);

			//Check if the line and plane are paralell, if not it hit.
			if (!MyMath::FloatEquals(dotLineAndPlaneNormal, 0.0f))
			{
				float t1 = Dot((planePos - origin), planeNormal) / dotLineAndPlaneNormal;

				if (t1 > 0.0f)
				{
					if (t1 < closest)
					{
						MyMath::Vector3 point = origin + (direction * t1);
						const float halfPlaneWidth = plane->GetWidth() * 0.5f;
						const float halfPlaneHeight = plane->GetHeight() * 0.5f;

						//If the ray hit inbetween the width & height.
						if (
							point.x > planePos.x - halfPlaneWidth && point.x < planePos.x + halfPlaneWidth &&	//Width
							point.z > planePos.z - halfPlaneHeight && point.z < planePos.z + halfPlaneHeight	//Height
							)
						{
							//1, 0, 0 is just temporary light direction.
							//#todo: INTRODUCE REAL LIGHTS!
							shadingValue = Dot(planeNormal, MyMath::Vector3(1.0f, 0.0f, 0.0f));

							//Comment in this if statement to get "backface" culling for planes.
							//if (shadingValue > 0.0f) {
							closest = t1;
							//}

							bestColor = plane->GetColor();
							bestNormal = planeNormal;

							//Reverse the normal if viewed from backside.
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
}

__global__
void RayTrace_ASCII(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingParameters* params,
	char* resultArray
)
{
	const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t column = blockIdx.x * blockDim.x + threadIdx.x;

	//Localization of variables.
	const size_t x = params->x;
	const size_t y = params->y;

	//x - 1 because the last x-line is for newlines.
	if (column >= (x - 1) || row >= y)
	{
		return;
	}

	//Set the amount of characters in the buffer per pixel depending on the mode.
	const size_t size = 12;

	//Convert pixel coordinates to (clip space? screen space?)
	const float convertedY = ((float)y - row * 2) / y;
	const float convertedX = (2 * column - (float)x) / x;

	//Localization of variables.
	const MyMath::Vector3 cameraPos = params->camPos;
	const float element1 = params->element1;
	const float element2 = params->element2;
	const MyMath::Matrix inverseVMatrix = params->inverseVMatrix;
	const float camFarDist = params->camFarDist;

	//Calculate the ray.
	const MyMath::Vector4 pixelVSpace = MyMath::Vector4(convertedX * element1, convertedY * element2, 1.0f, 0.0f);
	const MyMath::Vector3 directionWSpace = inverseVMatrix.Mult(pixelVSpace).xyz().Normalize_InPlace_GPU();

	TraceData traceData;
	Trace(directionWSpace, cameraPos, count, objects, traceData);

	//Decide what character to write for this pixel.
	char data = GetASCIICharacter(traceData.distance, camFarDist, traceData.shadingValue);
	
	//If the pixel hit something during ray tracing.
	if (data != ' ')
	{
		if (traceData.shadingValue < AMBIENT_LIGHT)
		{
			traceData.shadingValue = AMBIENT_LIGHT;
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

		char finalData[12] = {
			'\x1b', '[',			//Escape character
			'3', '8', ';',			//Keycode for foreground
			'5', ';',				//Keycode for foreground
			first, second, third,	//Index
			'm', data				//Character data.
		};
		memcpy(resultArray + (row * (x * size) + column * size), finalData, sizeof(char) * size);
		
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

__global__
void RayTrace_PIXEL(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingParameters* params,
	char* resultArray
)
{
	const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t column = blockIdx.x * blockDim.x + threadIdx.x;

	//Localization of variables.
	const size_t x = params->x;
	const size_t y = params->y;

	//x - 1 because the last x-line is for newlines.
	if (column >= (x - 1) || row >= y)
	{
		return;
	}

	//Set the amount of characters in the buffer per pixel depending on the mode.
	const size_t size = 12;

	//Convert pixel coordinates to (clip space? screen space?)
	const float convertedY = ((float)y - row * 2) / y;
	const float convertedX = (2 * column - (float)x) / x;

	//Localization of variables.
	const MyMath::Vector3 cameraPos = params->camPos;
	const float element1 = params->element1;
	const float element2 = params->element2;
	const MyMath::Matrix inverseVMatrix = params->inverseVMatrix;
	const float camFarDist = params->camFarDist;

	//Calculate the ray.
	const MyMath::Vector4 pixelVSpace = MyMath::Vector4(convertedX * element1, convertedY * element2, 1.0f, 0.0f);
	const MyMath::Vector3 directionWSpace = inverseVMatrix.Mult(pixelVSpace).xyz().Normalize_InPlace_GPU();

	TraceData traceData;
	Trace(directionWSpace, cameraPos, count, objects, traceData);

	//If the pixel hit something during ray tracing.
	if (data != ' ')
	{
		if (traceData.shadingValue < AMBIENT_LIGHT)
		{
			traceData.shadingValue = AMBIENT_LIGHT;
		}
		//Apply shading.
		bestColor *= traceData.shadingValue;

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

		char finalData[12] = {
				'\x1b', '[',			//Escape character
				'4', '8', ';',			//Keycode for background
				'5', ';',				//Keycode for background
				first, second, third,	//Index
				'm', ' '				//Character data.
		};
		memcpy(resultArray + (row * (x * size) + column * size), finalData, sizeof(char) * size);
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

__global__
void RayTrace_RGB_ASCII(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingParameters* params,
	char* resultArray
)
{
	const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t column = blockIdx.x * blockDim.x + threadIdx.x;

	//Localization of variables.
	const size_t x = params->x;
	const size_t y = params->y;

	//x - 1 because the last x-line is for newlines.
	if (column >= (x - 1) || row >= y)
	{
		return;
	}

	//Set the amount of characters in the buffer per pixel depending on the mode.
	const size_t size = 20;

	//Convert pixel coordinates to (clip space? screen space?)
	const float convertedY = ((float)y - row * 2) / y;
	const float convertedX = (2 * column - (float)x) / x;

	//Localization of variables.
	const MyMath::Vector3 cameraPos = params->camPos;
	const float element1 = params->element1;
	const float element2 = params->element2;
	const MyMath::Matrix inverseVMatrix = params->inverseVMatrix;
	const float camFarDist = params->camFarDist;

	//Calculate the ray.
	const MyMath::Vector4 pixelVSpace = MyMath::Vector4(convertedX * element1, convertedY * element2, 1.0f, 0.0f);
	const MyMath::Vector3 directionWSpace = inverseVMatrix.Mult(pixelVSpace).xyz().Normalize_InPlace_GPU();

	TraceData traceData;
	Trace(directionWSpace, cameraPos, count, objects, traceData);

	//Decide what character to write for this pixel.
	char data = GetASCIICharacter(traceData.distance, camFarDist, traceData.shadingValue);

	//If the pixel hit something during ray tracing.
	if (data != ' ')
	{
		if (traceData.shadingValue < AMBIENT_LIGHT)
		{
			traceData.shadingValue = AMBIENT_LIGHT;
		}
		//Apply shading.
		bestColor *= traceData.shadingValue;

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

		uint8_t originalIndex;
		uint8_t index;

		//R
		originalIndex = (uint8_t)bestColor.x;
		index = (uint8_t)bestColor.x;

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
		originalIndex = (uint8_t)bestColor.y;
		index = (uint8_t)bestColor.y;

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
		originalIndex = (uint8_t)bestColor.z;
		index = (uint8_t)bestColor.z;

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

		char finalData[20] = {
			'\x1b', '[',					//Escape character
			'3', '8', ';',					//Keycode for foreground
			'2', ';',						//Keycode for foreground
			firstR, secondR, thirdR, ';',	//R
			firstG, secondG, thirdG, ';',	//G
			firstB, secondB, thirdB,		//B
			'm', data						//Character data.
		};
		memcpy(resultArray + (row * (x * size) + column * size), finalData, sizeof(char) * size);
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
		memcpy(resultArray + (row * (x * size) + column * size), finalData, sizeof(char) * size);
	}
}

__global__
void RayTrace_RGB_PIXEL(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingParameters* params,
	char* resultArray
)
{
	const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t column = blockIdx.x * blockDim.x + threadIdx.x;

	//Localization of variables.
	const size_t x = params->x;
	const size_t y = params->y;

	//x - 1 because the last x-line is for newlines.
	if (column >= (x - 1) || row >= y)
	{
		return;
	}

	//Set the amount of characters in the buffer per pixel depending on the mode.
	const size_t size = 20;

	//Convert pixel coordinates to (clip space? screen space?)
	const float convertedY = ((float)y - row * 2) / y;
	const float convertedX = (2 * column - (float)x) / x;

	//Localization of variables.
	const MyMath::Vector3 cameraPos = params->camPos;
	const float element1 = params->element1;
	const float element2 = params->element2;
	const MyMath::Matrix inverseVMatrix = params->inverseVMatrix;
	const float camFarDist = params->camFarDist;

	//Calculate the ray.
	const MyMath::Vector4 pixelVSpace = MyMath::Vector4(convertedX * element1, convertedY * element2, 1.0f, 0.0f);
	const MyMath::Vector3 directionWSpace = inverseVMatrix.Mult(pixelVSpace).xyz().Normalize_InPlace_GPU();

	TraceData traceData;
	Trace(directionWSpace, cameraPos, count, objects, traceData);

	//If the pixel hit something during ray tracing.
	if (data != ' ')
	{
		if (traceData.shadingValue < AMBIENT_LIGHT)
		{
			traceData.shadingValue = AMBIENT_LIGHT;
		}
		//Apply shading.
		bestColor *= traceData.shadingValue;

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

		uint8_t originalIndex;
		uint8_t index;

		//R
		originalIndex = (uint8_t)bestColor.x;
		index = (uint8_t)bestColor.x;

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
		originalIndex = (uint8_t)bestColor.y;
		index = (uint8_t)bestColor.y;

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
		originalIndex = (uint8_t)bestColor.z;
		index = (uint8_t)bestColor.z;

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
		memcpy(resultArray + (row * (x * size) + column * size), finalData, sizeof(char) * size);
	}
}

__global__
void RayTrace_RGB_NORMALS(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingParameters* params,
	char* resultArray
)
{
	const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t column = blockIdx.x * blockDim.x + threadIdx.x;

	//Localization of variables.
	const size_t x = params->x;
	const size_t y = params->y;

	//x - 1 because the last x-line is for newlines.
	if (column >= (x - 1) || row >= y)
	{
		return;
	}

	//Set the amount of characters in the buffer per pixel depending on the mode.
	const size_t size = 20;

	//Convert pixel coordinates to (clip space? screen space?)
	const float convertedY = ((float)y - row * 2) / y;
	const float convertedX = (2 * column - (float)x) / x;

	//Localization of variables.
	const MyMath::Vector3 cameraPos = params->camPos;
	const float element1 = params->element1;
	const float element2 = params->element2;
	const MyMath::Matrix inverseVMatrix = params->inverseVMatrix;
	const float camFarDist = params->camFarDist;

	//Calculate the ray.
	const MyMath::Vector4 pixelVSpace = MyMath::Vector4(convertedX * element1, convertedY * element2, 1.0f, 0.0f);
	const MyMath::Vector3 directionWSpace = inverseVMatrix.Mult(pixelVSpace).xyz().Normalize_InPlace_GPU();

	TraceData traceData;
	Trace(directionWSpace, cameraPos, count, objects, traceData);

	//If the pixel hit something during ray tracing.
	if (data != ' ')
	{
		if (traceData.shadingValue < AMBIENT_LIGHT)
		{
			traceData.shadingValue = AMBIENT_LIGHT;
		}
		//Apply shading.
		bestColor *= traceData.shadingValue;

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

		
		uint8_t originalIndex;
		uint8_t index;

		//R
		originalIndex = (uint8_t)(bestNormal.x * 255);
		index = (uint8_t)(bestNormal.x * 255);

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
		originalIndex = (uint8_t)(bestNormal.y * 255);
		index = (uint8_t)(bestNormal.y * 255);

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
		originalIndex = (uint8_t)(bestNormal.z * 255);
		index = (uint8_t)(bestNormal.z * 255);

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
		memcpy(resultArray + (row * (x * size) + column * size), finalData, sizeof(char) * size);
	}
}

__global__
void RayTrace_SDL(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingParameters* params,
	char* resultArray
)
{
	const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t column = blockIdx.x * blockDim.x + threadIdx.x;

	//Localization of variables.
	const size_t x = params->x;
	const size_t y = params->y;

	//Convert pixel coordinates to (clip space? screen space?)
	const float convertedY = ((float)y - row * 2) / y;
	const float convertedX = (2 * column - (float)x) / x;

	//Localization of variables.
	const MyMath::Vector3 cameraPos = params->camPos;
	const float element1 = params->element1;
	const float element2 = params->element2;
	const MyMath::Matrix inverseVMatrix = params->inverseVMatrix;
	const float camFarDist = params->camFarDist;

	//Calculate the ray.
	const MyMath::Vector4 pixelVSpace = MyMath::Vector4(convertedX * element1, convertedY * element2, 1.0f, 0.0f);
	const MyMath::Vector3 directionWSpace = inverseVMatrix.Mult(pixelVSpace).xyz().Normalize_InPlace_GPU();

	TraceData traceData;
	Trace(directionWSpace, cameraPos, count, objects, traceData);
}

void RayTracing::RayTrace(Object3D* DEVICE_MEMORY_PTR const objects, const unsigned int count, const RayTracingParameters* params, char* resultArray, const RayTracingManager::RenderingMode mode)
{
	//Use different raytracing functions depending on the rendering mode.
	switch (mode)
	{
	case RayTracingManager::RenderingMode::ASCII:
		RayTrace_ASCII CUDA_KERNEL(gridDims, blockDims)(
			objects,
			count,
			params,
			resultArray);

		break;

	case RayTracingManager::RenderingMode::PIXEL:
		RayTrace_PIXEL CUDA_KERNEL(gridDims, blockDims)(
			objects,
			count,
			params,
			resultArray);

		break;

	case RayTracingManager::RenderingMode::RGB_ASCII:
		RayTrace_RGB_ASCII CUDA_KERNEL(gridDims, blockDims)(
			objects,
			count,
			params,
			resultArray);

		break;

	case RayTracingManager::RenderingMode::RGB_PIXEL:
		RayTrace_RGB_PIXEL CUDA_KERNEL(gridDims, blockDims)(
			objects,
			count,
			params,
			resultArray);

		break;

	case RayTracingManager::RenderingMode::RGB_NORMALS:
		RayTrace_RGB_NORMALS CUDA_KERNEL(gridDims, blockDims)(
			objects,
			count,
			params,
			resultArray);

		break;

	case RayTracingManager::RenderingMode::SDL:
		RayTrace_SDL CUDA_KERNEL(gridDims, blockDims)(
			objects,
			count,
			params,
			resultArray);

		break;

	default:
		assert(false && "Invalid rendering mode.");
		break;
	}
}
