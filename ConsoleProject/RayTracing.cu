#include "pch.h"
#include "RayTracing.h"

#include "ANSIRGB.h"

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

__global__ void RayTrace(
	Object3D* DEVICE_MEMORY_PTR const objects,
	const unsigned int count,
	const RayTracingParameters* params,
	char* resultArray,
	const PrintMachine::PrintMode mode
)
{
	const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t column = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Localization of variables.
	const size_t x = params->x;
	const size_t y = params->y;

	if (column >= x || row >= y)
	{
		return;
	}

	//Set the amount of characters in the buffer per pixel depending on the mode.
	const size_t size = (mode == PrintMachine::ASCII || mode == PrintMachine::PIXEL) ? 12 : 20;

	//#todo: Do this when minimizing instead of in a thread.
	//If the pixel is at the end of a line, output \n and return.
	if (column == (x - 1))
	{
		resultArray[row * (x * size) + column * size] = '\n';
		return;
	}

	//Convert pixel coordinates to (clip space? screen space?)
	const float convertedY = ((float)y - row * 2) / y;
	const float convertedX = (2 * column - (float)x) / x;

	//Localization of variables.
	MyMath::Vector3 cameraPos = params->camPos;
	const float element1 = params->element1;
	const float element2 = params->element2;
	const MyMath::Matrix inverseVMatrix = params->inverseVMatrix;
	const float camFarDist = params->camFarDist;

	//Calculate the ray.
	const MyMath::Vector4 pixelVSpace = MyMath::Vector4(convertedX * element1, convertedY * element2, 1.0f, 0.0f);
	const MyMath::Vector3 directionWSpace = inverseVMatrix.Mult(pixelVSpace).xyz().Normalize_InPlace();
	
	//Used during intersection tests with spheres.
	const float a = Dot(directionWSpace, directionWSpace);
	const float fourA = 4.0f * a;
	const float divTwoA = 1.0f / (2.0f * a);

	char data = ' ';
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

			const MyMath::Vector3 objectToCam = cameraPos - spherePos;
			const float radius = sphere->GetRadius();

			const float b = 2.0f * Dot(directionWSpace, objectToCam);
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
					const MyMath::Vector3 normalSphere = (cameraPos + directionWSpace * closest - spherePos).Normalize();
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

			const float dotLineAndPlaneNormal = Dot(directionWSpace, planeNormal);

			//Check if the line and plane are paralell, if not it hit.
			if (!MyMath::FloatEquals(dotLineAndPlaneNormal, 0.0f))
			{
				float t1 = Dot((planePos - cameraPos), planeNormal) / dotLineAndPlaneNormal;

				if (t1 > 0.0f)
				{
					if (t1 < closest)
					{
						MyMath::Vector3 point = cameraPos + (directionWSpace * t1);
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

	//Decide what character to write for this pixel.
	static const float t = 67;
	//If we miss or its outside the frustum we dont print anything.
	if (closest > camFarDist)
	{
		data = ascii[0];
	}
	else
	{
		const int dataIndex = MyMath::Clamp((int)ceil(shadingValue * t), 1, NUM_ASCII_CHARACTERS); //Clamp with 1 as min, so that the empty space does not get used.
		data = ascii[dataIndex];
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

RayTracer::RayTracer()
{
	const size_t size = PrintMachine::GetMaxSize();

	cudaMalloc(&m_deviceResultArray, sizeof(char) * size);

	//Allocate the array which will contain the full screen before minimization.
	m_hostResultArray = std::make_unique<char[]>(size);

	//Allocate the minimized array which will be printed to the console.
	m_minimizedResultArray = std::make_unique<char[]>(size);
}

RayTracer::~RayTracer()
{
	cudaFree(m_deviceResultArray);
}

void RayTracer::RayTracingWrapper(
	const size_t x,
	const size_t y,
	const DeviceObjectArray<Object3D*>& deviceObjects,
	const RayTracingParameters DEVICE_MEMORY_PTR rayTracingParameters,
	double dt
)
{
	//The backbuffer needs to be reset in order to not produce artefacts, especially when switching printing mode to RGB.
	ResetDeviceBackBuffer();

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
		deviceObjects.m_deviceArray,
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
	gridDims.x = static_cast<unsigned int>(std::ceil((x + 1) / 16.0));
	gridDims.y = static_cast<unsigned int>(std::ceil(y / 16.0));
	blockDims.x = 16u;
	blockDims.y = 16u;
	
	RayTrace<<<gridDims, blockDims>>>(
		deviceObjects.m_deviceArray,
		deviceObjects.count,
		rayTracingParameters,
		m_deviceResultArray,
		PrintMachine::GetPrintMode()
	);
	//Make sure all the threads are done with the ray tracing.
	gpuErrchk(cudaDeviceSynchronize());

	//#todo: Make a function to get the original "max size". Why?
	const size_t size = PrintMachine::GetMaxSize();

	//Copy all data from GPU -> CPU.
	gpuErrchk(cudaMemcpy(m_hostResultArray.get(), m_deviceResultArray, size, cudaMemcpyDeviceToHost));

	//Minimize the result, by removing unneccessary ANSI escape sequences.
	size_t newSize = MinimizeResults(size, y);

	//Locking/unlocking of the mutex, flagging for changing buffer, and changing the printing size now all happens within this function.
	//-----------------------------------------------------------------------------------------------------
	PrintMachine::SetDataInBackBuffer(m_minimizedResultArray.get(), newSize);
	//-----------------------------------------------------------------------------------------------------

	return;
}

void RayTracer::ResetDeviceBackBuffer()
{
	const size_t size = PrintMachine::GetMaxSize();
	cudaMemset(m_deviceResultArray, 0, size);
}

size_t RayTracer::MinimizeResults(const size_t size, const size_t y)
{
	PrintMachine::PrintMode mode = PrintMachine::GetPrintMode();


	//If its in 8 bit mode.
	if (mode == PrintMachine::ASCII || mode == PrintMachine::PIXEL)
	{
		return Minimize8bit(size, y);
	}
	//Else it is rgb.
	else
	{
		return MinimizeRGB(size, y);
	}
}

size_t RayTracer::Minimize8bit(const size_t size, const size_t y)
{
	size_t newlines = 0;
	size_t addedChars = 0;

	//We hold a pointer to the spot in the buffer with the latest color.
	char* latestColor = nullptr;

	for (size_t i = 0; i < size;)
	{
		char current = m_hostResultArray[i];

		//If we are handling a pixel.
		if (current == '\x1b')
		{
			//If its not the same color add the whole escape sequence and update latest color.
			if (
				!latestColor ||
				latestColor[0] != m_hostResultArray[i + 7] ||
				latestColor[1] != m_hostResultArray[i + 8] ||
				latestColor[2] != m_hostResultArray[i + 9]
			)
			{
				//Move the pointer to the spot in the array with the color.
				latestColor = m_hostResultArray.get() + i + 7;

				//Copy the escape sequence and data to the minimized result.
				memcpy(m_minimizedResultArray.get() + addedChars, m_hostResultArray.get() + i, 12);

				addedChars += 12;
			}
			//Only add the data and not the escape sequence.
			else
			{
				m_minimizedResultArray[addedChars] = m_hostResultArray[i + 11];

				addedChars += 1;
			}

			//Move 12 characters forward.
			i += 12;
		}
		//If we are handling the end of a line.
		else if (current == '\n')
		{
			++newlines;

			m_minimizedResultArray[addedChars] = '\n';
			++addedChars;

			//Move 1 character forward.
			++i;

			//Stop iterating if the amount of lines equals the height.
			if (newlines == y)
			{
				break;
			}
		}
		//For \0. Simply move one character forward.
		else
		{
			++i;
		}
	}

	return addedChars;
}

size_t RayTracer::MinimizeRGB(const size_t size, const size_t y)
{
	size_t newlines = 0;
	size_t addedChars = 0;

	//We hold a pointer to the spot in the buffer with the latest color.
	char* latestColor = nullptr;

	for (size_t i = 0; i < size;)
	{
		char current = m_hostResultArray[i];

		//If we are handling a pixel.
		if (current == '\x1b')
		{
			//If its not the same color Add the escape sequence and update latest color.
			if (
				!latestColor ||
				latestColor[0] != m_hostResultArray[i + 7] || latestColor[1] != m_hostResultArray[i + 8] || latestColor[2] != m_hostResultArray[i + 9] ||		//R
				latestColor[4] != m_hostResultArray[i + 11] || latestColor[5] != m_hostResultArray[i + 12] || latestColor[6] != m_hostResultArray[i + 13] ||	//G
				latestColor[8] != m_hostResultArray[i + 15] || latestColor[9] != m_hostResultArray[i + 16] || latestColor[10] != m_hostResultArray[i + 17]		//B
			)
			{
				//Move the pointer to the spot in the array with the color.
				latestColor = m_hostResultArray.get() + i + 7;

				//Copy the escape sequence and data to the minimized result.
				memcpy(m_minimizedResultArray.get() + addedChars, m_hostResultArray.get() + i, 20);

				addedChars += 20;
			}
			//Only add the data and not the escape sequence.
			else
			{
				m_minimizedResultArray[addedChars] = m_hostResultArray[i + 19];

				addedChars += 1;
			}

			//Move 20 characters forward.
			i += 20;
		}
		//If we are handling the end of a line.
		else if (current == '\n')
		{
			++newlines;

			m_minimizedResultArray[addedChars] = m_hostResultArray[i];
			++addedChars;

			//Move 1 character forward.
			++i;

			//Stop iterating if the amount of lines equals the height.
			if (newlines == PrintMachine::GetHeight())
			{
				break;
			}
		}
		//For \0. Simply move 1 character forward.
		else
		{
			++i;
		}
	}

	return addedChars;
}
