#include "pch.h"
#include "RayTracing.h"

#include "ANSIRGB.h"
#include "RayTracingManager.h"
#include "Sphere.h"
#include "Plane.h"

__device__
MyMath::Vector3 CalculateInitialDirection(const RayTracingCPUToGPUData* params)
{
	const size_t row = blockIdxY * blockDimY + threadIdxY;
	const size_t column = blockIdxX * blockDimX + threadIdxX;

	//Convert pixel coordinates to (clip space? screen space?)
	const float convertedY = ((float)(params->y) - row * 2) / params->y;
	const float convertedX = (2 * column - (float)(params->x)) / params->x;

	//Calculate the ray.
	const MyMath::Vector4 pixelVSpace = MyMath::Vector4(convertedX * params->element1, convertedY * params->element2, 1.0f, 0.0f);

	const MyMath::Vector3 WorldSpaceDirection = params->inverseVMatrix.Mult(pixelVSpace).xyz();
	return WorldSpaceDirection.Normalize_GPU();
}

__device__
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

__device__ MyMath::Vector3 BlinnPhongShading(
	const MyMath::Vector3& objectDiffuseColour, const MyMath::Vector3& objectSpecularColour,
	const MyMath::Vector3& lightPos,
	const MyMath::Vector3& lightDiffuseColour, const float lightDiffusePower,
	const MyMath::Vector3& lightSpecularColour, const float lightSpecularPower,
	const MyMath::Vector3& point, const MyMath::Vector3& viewDir, const MyMath::Vector3& normal)
{
	MyMath::Vector3 lightDir = (lightPos - point); //Vector from the point to the light.

	float distance = lightDir.Length();
	distance = distance * distance; //Square the distance for attenuation.
	const float divDistance = 1.0f / distance; //Precalculate 1/distance for optimization.

	lightDir = lightDir.Normalize_GPU();

	MyMath::Vector3 normalizedNormal = normal.Normalize_GPU();
	MyMath::Vector3 normalizedViewDir = viewDir.Normalize_GPU();

	//Intensity of the diffuse light. Clamp to keep within the 0-1 range.
	float NdotL = MyMath::Dot(normalizedNormal, lightDir);
	float diffuseIntensity = MyMath::Clamp(NdotL, 0.0f, 1.0f);

	// Calculate the diffuse light factoring in light color, power and the attenuation
	const MyMath::Vector3 diffuse = lightDiffuseColour * diffuseIntensity * lightDiffusePower * divDistance;

	//Calculate the half vector between the light vector and the view vector.
	const MyMath::Vector3 h = (lightDir + normalizedViewDir).Normalize_GPU();

	static constexpr float specularHardness = 32.0f;

	//Intensity of the specular light
	const float NdotH = MyMath::Dot(normalizedNormal, h);
	const float specularIntensity = pow(MyMath::Clamp(NdotH, 0.0f, 1.0f), specularHardness); //Hardcoded shininess.

	const MyMath::Vector3 specular = lightSpecularColour * specularIntensity * lightSpecularPower * divDistance;

	const MyMath::Vector3 ambientLight(0.2f, 0.2f, 0.2f);
	return MyMath::ComponentMul(ambientLight, objectDiffuseColour) + MyMath::ComponentMul(diffuse, objectDiffuseColour) + MyMath::ComponentMul(specular, objectSpecularColour);
}

__device__
void RayTrace(
	const RayTraceInputData& rayTraceInputData,
	RayTraceReturnData& rayTraceReturnData)
{
	ObjectTraceInputData objectTraceInputData;
	objectTraceInputData.origin = rayTraceInputData.origin;
	objectTraceInputData.direction = rayTraceInputData.direction;

	//Used during intersection tests with spheres.
	objectTraceInputData.a = MyMath::Dot(rayTraceInputData.direction, rayTraceInputData.direction);
	objectTraceInputData.fourA = 4.0f * objectTraceInputData.a;
	objectTraceInputData.divTwoA = 1.0f / (2.0f * objectTraceInputData.a);

	ObjectTraceReturnData objectTraceReturnData;

	bool bHitSomething = false;

	//Ray trace against every object.
	for (size_t i = 0; i < rayTraceInputData.objectCount; i++)
	{
		//#todo: Here we need to check if the object is culled, if it is we continue on the next object.

		const ObjectType type = rayTraceInputData.objects[i]->GetType();
		switch (type)
		{
		case ObjectType::PlaneType:
		{
			const Plane* plane = (Plane*)(rayTraceInputData.objects[i]);
			plane->Trace(objectTraceInputData, objectTraceReturnData);
			break;
		}
		case ObjectType::SphereType:
		{
			const Sphere* sphere = (Sphere*)(rayTraceInputData.objects[i]);
			sphere->Trace(objectTraceInputData, objectTraceReturnData);
			break;
		}
		default:
			break;
		}

		if (objectTraceReturnData.bHit && objectTraceReturnData.distance < rayTraceReturnData.distance)
		{
			bHitSomething = true;

			rayTraceReturnData.distance = objectTraceReturnData.distance;
			rayTraceReturnData.normal = objectTraceReturnData.normal;
			rayTraceReturnData.normal = rayTraceReturnData.normal.Normalize_GPU();

			//1, 0, 0 is just temporary light direction.
			//#todo: INTRODUCE REAL LIGHTS!
			rayTraceReturnData.shadingValue = Dot(rayTraceReturnData.normal, MyMath::Vector3(1.0f, 0.0f, 0.0f));
			rayTraceReturnData.color = rayTraceInputData.objects[i]->GetColor();
		}
	}

	if (!bHitSomething)
	{
		return;
	}

	MyMath::Vector3 shading = BlinnPhongShading(
		rayTraceReturnData.color / 255.0f, //Object diffuse color
		MyMath::Vector3(1.0f, 1.0f, 1.0f), //Object specular color
		MyMath::Vector3(1.0f, 50.0f, 0.0f), //Light position.
		MyMath::Vector3(1.0f, 1.0f, 1.0f), 2000.0f, //Diffuse color and power of the light.
		MyMath::Vector3(1.0f, 1.0f, 1.0f), 3000.0f, //Specular color and power of the light.
		rayTraceInputData.origin + rayTraceInputData.direction * rayTraceReturnData.distance, //Point in world space.
		(rayTraceInputData.direction * -1.0f).Normalize_GPU(), //View direction.
		rayTraceReturnData.normal //Normal at the point.
	);

	shading *= 255.0f; //Scale to 0-255 range.

	//rayTraceReturnData.color = MyMath::ComponentMul(rayTraceReturnData.color, shading);
	rayTraceReturnData.color = MyMath::Vector3(MyMath::Min(255.0f, shading.x), MyMath::Min(255.0f, shading.y), MyMath::Min(255.0f, shading.z));
	//rayTraceReturnData.color += shading;
	/*
	if (rayTraceReturnData.shadingValue < AMBIENT_LIGHT)
	{
		rayTraceReturnData.shadingValue = AMBIENT_LIGHT;
	}

	//#TODO: THIS SHOULD BE DONE WITH PHONG SHADING INSTEAD!
	//Apply shading.
	rayTraceReturnData.color *= rayTraceReturnData.shadingValue;*/
}

__global__
void RayTrace_ASCII(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingCPUToGPUData* params,
	char* resultArray
)
{
	const size_t row = blockIdxY * blockDimY + threadIdxY;
	const size_t column = blockIdxX * blockDimX + threadIdxX;

	//Localization of parameters.
	RayTracingCPUToGPUData localParams;

	localParams = *params;

	//x - 1 because the last x-line is for newlines.
	if (column >= (localParams.x - 1) || row >= localParams.y)
	{
		return;
	}

	const MyMath::Vector3 directionWSpace = CalculateInitialDirection(params);

	RayTraceInputData rayTraceInputData;
	rayTraceInputData.origin = localParams.camPos;
	rayTraceInputData.direction = directionWSpace;
	rayTraceInputData.objectCount = count;
	rayTraceInputData.objects = objects;

	RayTraceReturnData rayTraceData;
	RayTrace(rayTraceInputData, rayTraceData);

	//Decide what character to write for this pixel.
	const char data = GetASCIICharacter(rayTraceData.distance, localParams.camFarDist, rayTraceData.shadingValue);
	
	//If the pixel hit something during ray tracing.
	if (rayTraceData.distance <= localParams.camFarDist)
	{
		//Convert the 24bit RGB color to ANSI 8 bit color.
		uint8_t index = ansi256_from_rgb(((uint8_t)rayTraceData.color.x << 16) + ((uint8_t)rayTraceData.color.y << 8) + (uint8_t)rayTraceData.color.z);
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

		char finalData[SIZE_8BIT] = {
			'\x1b', '[',			//Escape character
			'3', '8', ';',			//Keycode for foreground
			'5', ';',				//Keycode for foreground
			first, second, third,	//Index
			'm', data				//Character data.
		};
		memcpy(resultArray + (row * (localParams.x * SIZE_8BIT) + column * SIZE_8BIT), finalData, sizeof(char) * SIZE_8BIT);
		
	}
	//If it is an empty space we can not use a background color.
	else
	{
		char finalData[SIZE_8BIT] = {
			'\x1b', '[',			//Escape character
			'4', '8', ';',			//Keycode for background
			'5', ';',				//Keycode for background
			'\0', '1', '6',			//Index
			'm', ' '				//Character data.
		};
		memcpy(resultArray + (row * (localParams.x * SIZE_8BIT) + column * SIZE_8BIT), finalData, sizeof(char) * SIZE_8BIT);
	}
}

__global__
void RayTrace_PIXEL(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingCPUToGPUData* params,
	char* resultArray
)
{
	const size_t row = blockIdxY * blockDimY + threadIdxY;
	const size_t column = blockIdxX * blockDimX + threadIdxX;

	//Localization of parameters.
	RayTracingCPUToGPUData localParams;
	localParams = *params;

	//x - 1 because the last x-line is for newlines.
	if (column >= (localParams.x - 1) || row >= localParams.y)
	{
		return;
	}

	const MyMath::Vector3 directionWSpace = CalculateInitialDirection(params);

	RayTraceInputData rayTraceInputData;
	rayTraceInputData.origin = localParams.camPos;
	rayTraceInputData.direction = directionWSpace;
	rayTraceInputData.objectCount = count;
	rayTraceInputData.objects = objects;

	RayTraceReturnData rayTraceData;
	RayTrace(rayTraceInputData, rayTraceData);

	//If the pixel hit something during ray tracing.
	if (rayTraceData.distance <= localParams.camFarDist)
	{
		//Convert the 24bit RGB color to ANSI 8 bit color.
		uint8_t index = ansi256_from_rgb(((uint8_t)rayTraceData.color.x << 16) + ((uint8_t)rayTraceData.color.y << 8) + (uint8_t)rayTraceData.color.z);
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

		char finalData[SIZE_8BIT] = {
				'\x1b', '[',			//Escape character
				'4', '8', ';',			//Keycode for background
				'5', ';',				//Keycode for background
				first, second, third,	//Index
				'm', ' '				//Character data.
		};
		memcpy(resultArray + (row * (localParams.x * SIZE_8BIT) + column * SIZE_8BIT), finalData, sizeof(char) * SIZE_8BIT);
	}
	//If it is an empty space we can not use a background color.
	else
	{
		char finalData[SIZE_8BIT] = {
			'\x1b', '[',			//Escape character
			'4', '8', ';',			//Keycode for background
			'5', ';',				//Keycode for background
			'\0', '1', '6',			//Index
			'm', ' '				//Character data.
		};
		memcpy(resultArray + (row * (localParams.x * SIZE_8BIT) + column * SIZE_8BIT), finalData, sizeof(char) * SIZE_8BIT);
	}
}

__global__
void RayTrace_RGB_ASCII(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingCPUToGPUData* params,
	char* resultArray
)
{
	const size_t row = blockIdxY * blockDimY + threadIdxY;
	const size_t column = blockIdxX * blockDimX + threadIdxX;

	//Localization of parameters.
	RayTracingCPUToGPUData localParams;
	localParams = *params;

	//x - 1 because the last x-line is for newlines.
	if (column >= (localParams.x - 1) || row >= localParams.y)
	{
		return;
	}

	const MyMath::Vector3 directionWSpace = CalculateInitialDirection(params);

	RayTraceInputData rayTraceInputData;
	rayTraceInputData.origin = localParams.camPos;
	rayTraceInputData.direction = directionWSpace;
	rayTraceInputData.objectCount = count;
	rayTraceInputData.objects = objects;

	RayTraceReturnData rayTraceData;
	RayTrace(rayTraceInputData, rayTraceData);

	//Decide what character to write for this pixel.
	const char data = GetASCIICharacter(rayTraceData.distance, localParams.camFarDist, rayTraceData.shadingValue);

	//If the pixel hit something during ray tracing.
	if (rayTraceData.distance <= localParams.camFarDist)
	{
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
		originalIndex = (uint8_t)rayTraceData.color.x;
		index = (uint8_t)rayTraceData.color.x;

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
		originalIndex = (uint8_t)rayTraceData.color.y;
		index = (uint8_t)rayTraceData.color.y;

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
		originalIndex = (uint8_t)rayTraceData.color.z;
		index = (uint8_t)rayTraceData.color.z;

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

		char finalData[SIZE_RGB] = {
			'\x1b', '[',					//Escape character
			'3', '8', ';',					//Keycode for foreground
			'2', ';',						//Keycode for foreground
			firstR, secondR, thirdR, ';',	//R
			firstG, secondG, thirdG, ';',	//G
			firstB, secondB, thirdB,		//B
			'm', data						//Character data.
		};
		memcpy(resultArray + (row * (localParams.x * SIZE_RGB) + column * SIZE_RGB), finalData, sizeof(char) * SIZE_RGB);
	}
	//If it is an empty space we can not use a background color. 
	else
	{
		char finalData[SIZE_RGB] = {
			'\x1b', '[',			//Escape character
			'4', '8', ';',			//Keycode for background
			'2', ';',				//Keycode for background
			'\0', '\0', '0', ';',	//R
			'\0', '\0', '0', ';',	//G
			'\0', '\0', '0',		//B
			'm', ' '				//Character data.
		};
		memcpy(resultArray + (row * (localParams.x * SIZE_RGB) + column * SIZE_RGB), finalData, sizeof(char) * SIZE_RGB);
	}
}

__global__
void RayTrace_RGB_PIXEL(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingCPUToGPUData* params,
	char* resultArray
)
{
	const size_t row = blockIdxY * blockDimY + threadIdxY;
	const size_t column = blockIdxX * blockDimX + threadIdxX;

	//Localization of parameters.
	RayTracingCPUToGPUData localParams;
	localParams = *params;

	//x - 1 because the last x-line is for newlines.
	if (column >= (localParams.x - 1) || row >= localParams.y)
	{
		return;
	}

	const MyMath::Vector3 directionWSpace = CalculateInitialDirection(params);

	RayTraceInputData rayTraceInputData;
	rayTraceInputData.origin = localParams.camPos;
	rayTraceInputData.direction = directionWSpace;
	rayTraceInputData.objectCount = count;
	rayTraceInputData.objects = objects;

	RayTraceReturnData rayTraceData;
	RayTrace(rayTraceInputData, rayTraceData);

	//If the pixel hit something during ray tracing.
	if (rayTraceData.distance <= localParams.camFarDist)
	{
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
		originalIndex = (uint8_t)rayTraceData.color.x;
		index = (uint8_t)rayTraceData.color.x;

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
		originalIndex = (uint8_t)rayTraceData.color.y;
		index = (uint8_t)rayTraceData.color.y;

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
		originalIndex = (uint8_t)rayTraceData.color.z;
		index = (uint8_t)rayTraceData.color.z;

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

		char finalData[SIZE_RGB] = {
			'\x1b', '[',					//Escape character
			'4', '8', ';',					//Keycode for foreground
			'2', ';',						//Keycode for foreground
			firstR, secondR, thirdR, ';',	//R
			firstG, secondG, thirdG, ';',	//G
			firstB, secondB, thirdB,		//B
			'm', ' '						//Character data.
		};
		memcpy(resultArray + (row * (localParams.x * SIZE_RGB) + column * SIZE_RGB), finalData, sizeof(char) * SIZE_RGB);
	}
	//If it is an empty space we can not use a background color. 
	else
	{
		char finalData[SIZE_RGB] = {
			'\x1b', '[',			//Escape character
			'4', '8', ';',			//Keycode for background
			'2', ';',				//Keycode for background
			'\0', '\0', '0', ';',	//R
			'\0', '\0', '0', ';',	//G
			'\0', '\0', '0',		//B
			'm', ' '				//Character data.
		};
		memcpy(resultArray + (row * (localParams.x * SIZE_RGB) + column * SIZE_RGB), finalData, sizeof(char) * SIZE_RGB);
	}
}

__global__
void RayTrace_RGB_NORMALS(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingCPUToGPUData* params,
	char* resultArray
)
{
	const size_t row = blockIdxY * blockDimY + threadIdxY;
	const size_t column = blockIdxX * blockDimX + threadIdxX;

	//Localization of parameters.
	RayTracingCPUToGPUData localParams;
	localParams = *params;

	//x - 1 because the last x-line is for newlines.
	if (column >= (localParams.x - 1) || row >= localParams.y)
	{
		return;
	}

	const MyMath::Vector3 directionWSpace = CalculateInitialDirection(params);

	//#TODO: Add a special trace function for normals, which is not recursive.
	RayTraceInputData rayTraceInputData;
	rayTraceInputData.origin = localParams.camPos;
	rayTraceInputData.direction = directionWSpace;
	rayTraceInputData.objectCount = count;
	rayTraceInputData.objects = objects;

	RayTraceReturnData rayTraceData;
	RayTrace(rayTraceInputData, rayTraceData);

	//If the pixel hit something during ray tracing.
	if (rayTraceData.distance <= localParams.camFarDist)
	{
		//Remove the shading, since we are working with normals.
		rayTraceData.color /= rayTraceData.shadingValue;

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
		originalIndex = (uint8_t)(rayTraceData.normal.x * 255);
		index = (uint8_t)(rayTraceData.normal.x * 255);

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
		originalIndex = (uint8_t)(rayTraceData.normal.y * 255);
		index = (uint8_t)(rayTraceData.normal.y * 255);

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
		originalIndex = (uint8_t)(rayTraceData.normal.z * 255);
		index = (uint8_t)(rayTraceData.normal.z * 255);

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

		char finalData[SIZE_RGB] = {
			'\x1b', '[',					//Escape character
			'4', '8', ';',					//Keycode for foreground
			'2', ';',						//Keycode for foreground
			firstR, secondR, thirdR, ';',	//R
			firstG, secondG, thirdG, ';',	//G
			firstB, secondB, thirdB,		//B
			'm', ' '						//Character data.
		};
		memcpy(resultArray + (row * (localParams.x * SIZE_RGB) + column * SIZE_RGB), finalData, sizeof(char) * SIZE_RGB);
	}
	//If it is an empty space we can not use a background color. 
	else
	{
		char finalData[SIZE_RGB] = {
			'\x1b', '[',			//Escape character
			'4', '8', ';',			//Keycode for background
			'2', ';',				//Keycode for background
			'\0', '\0', '0', ';',	//R
			'\0', '\0', '0', ';',	//G
			'\0', '\0', '0',		//B
			'm', ' '				//Character data.
		};
		memcpy(resultArray + (row * (localParams.x * SIZE_RGB) + column * SIZE_RGB), finalData, sizeof(char) * SIZE_RGB);
	}
}

__global__
void RayTrace_SDL(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingCPUToGPUData* params,
	char* resultArray
)
{
	const size_t row = blockIdxY * blockDimY + threadIdxY;
	const size_t column = blockIdxX * blockDimX + threadIdxX;

	//Localization of parameters.
	RayTracingCPUToGPUData localParams;
	localParams = *params;

	//x - 1 because the last x-line is for newlines.
	if (column >= (localParams.x - 1) || row >= localParams.y)
	{
		return;
	}

	const MyMath::Vector3 directionWSpace = CalculateInitialDirection(params);

	RayTraceInputData rayTraceInputData;
	rayTraceInputData.origin = localParams.camPos;
	rayTraceInputData.direction = directionWSpace;
	rayTraceInputData.objectCount = count;
	rayTraceInputData.objects = objects;

	RayTraceReturnData rayTraceData;
	RayTrace(rayTraceInputData, rayTraceData);

	//If the pixel hit something during ray tracing.
	if (rayTraceData.distance <= localParams.camFarDist)
	{

	}
	else
	{

	}
}

void RayTracing::RayTrace(
	const dim3& gridDims,
	const dim3& blockDims,
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingCPUToGPUData* params,
	char* resultArray,
	const RenderingMode mode)
{
	//Use different raytracing functions depending on the rendering mode.
	switch (mode)
	{
	case RenderingMode::BIT_ASCII:
		RayTrace_ASCII CUDA_KERNEL(gridDims, blockDims)(
			objects,
			count,
			params,
			resultArray);

		break;

	case RenderingMode::BIT_PIXEL:
		RayTrace_PIXEL CUDA_KERNEL(gridDims, blockDims)(
			objects,
			count,
			params,
			resultArray);

		break;

	case RenderingMode::RGB_ASCII:
		RayTrace_RGB_ASCII CUDA_KERNEL(gridDims, blockDims)(
			objects,
			count,
			params,
			resultArray);

		break;

	case RenderingMode::RGB_PIXEL:
		RayTrace_RGB_PIXEL CUDA_KERNEL(gridDims, blockDims)(
			objects,
			count,
			params,
			resultArray);

		break;

	case RenderingMode::RGB_NORMALS:
		RayTrace_RGB_NORMALS CUDA_KERNEL(gridDims, blockDims)(
			objects,
			count,
			params,
			resultArray);

		break;

	case RenderingMode::SDL:
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
