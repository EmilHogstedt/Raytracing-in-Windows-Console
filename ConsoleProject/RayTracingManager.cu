#include "pch.h"
#include "RayTracingManager.h"

#include "PrintMachine.h"
#include "Sphere.h"
#include "Plane.h"
#include "RayTracing.h"


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

RayTracingManager::RayTracingManager()
{
	//Allocate memory for the raytracingparameters.
	cudaMalloc(&m_deviceRayTracingParameters, sizeof(RayTracingParameters));

	const size_t size = PrintMachine::GetMaxSize();

	cudaMalloc(&m_deviceResultArray, sizeof(char) * size);

	//Allocate the array which will contain the full screen before minimization.
	m_hostResultArray = std::make_unique<char[]>(size);

	//Allocate the minimized array which will be printed to the console.
	m_minimizedResultArray = std::make_unique<char[]>(size);
}

RayTracingManager::~RayTracingManager()
{
	cudaFree(m_deviceResultArray);

	cudaFree(m_deviceRayTracingParameters);
}

void RayTracingManager::Update(
	const RayTracingParameters& params,
	const DeviceObjectArray<Object3D*>& deviceObjects,
	double dt
)
{
	//Set the raytracingparameters on the GPU.
	gpuErrchk(cudaMemcpy(m_deviceRayTracingParameters, &params, sizeof(RayTracingParameters), cudaMemcpyHostToDevice));

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
	UpdateObjects CUDA_KERNEL(gridDims, blockDims)(
		deviceObjects.m_deviceArray,
		deviceObjects.count,
		dt
	);

	//Classify the objects into the octtree.
	//Mark objects within the frustum
	/*
	Culling CUDA_KERNEL(gridDims, blockDims)(
		deviceObjects.using1st ? deviceObjects.m_deviceArray1 : deviceObjects.m_deviceArray2,

	);
	*/
	//After we do the culling we check the remaining objects within the octtree and update their closest position to the camera.


	//Do the raytracing. Calculate x and y dimensions in blocks depending on screensize.
	//1 thread per pixel.
	gridDims.x = static_cast<unsigned int>(std::ceil((params.x + 1) / 16.0));
	gridDims.y = static_cast<unsigned int>(std::ceil(params.y / 16.0));
	blockDims.x = 16u;
	blockDims.y = 16u;

	RayTracing::RayTrace(
		gridDims,
		blockDims,
		deviceObjects.m_deviceArray,
		deviceObjects.count,
		m_deviceRayTracingParameters,
		m_deviceResultArray,
		currentRenderingMode);

	//Make sure all the threads are done with the ray tracing.
	gpuErrchk(cudaDeviceSynchronize());

	//#todo: Make a function to get the original "max size". Why?
	const size_t size = PrintMachine::GetMaxSize();

	//Copy all data from GPU -> CPU.
	gpuErrchk(cudaMemcpy(m_hostResultArray.get(), m_deviceResultArray, size, cudaMemcpyDeviceToHost));

	//Minimize the result, by removing unneccessary ANSI escape sequences.
	size_t newSize = MinimizeResults(size, params.x, params.y);

	//Locking/unlocking of the mutex, flagging for changing buffer, and changing the printing size now all happens within this function.
	//-----------------------------------------------------------------------------------------------------
	PrintMachine::SetDataInBackBuffer(m_minimizedResultArray.get(), newSize);
	//-----------------------------------------------------------------------------------------------------

	return;
}

void RayTracingManager::SetRenderingMode(const RenderingMode newRenderMode)
{
	currentRenderingMode = newRenderMode;
}

void RayTracingManager::ResetDeviceBackBuffer()
{
	const size_t size = PrintMachine::GetMaxSize();
	cudaMemset(m_deviceResultArray, 0, size);
}

size_t RayTracingManager::MinimizeResults(const size_t size, const size_t x, const size_t y)
{
	//If its in 8 bit mode.
	if (currentRenderingMode == RenderingMode::BIT_ASCII)
	{
		return MinimizeASCII(size, x, y);
	}
	else if (currentRenderingMode == RenderingMode::BIT_PIXEL)
	{
		return MinimizePIXEL(size, x, y);
	}
	//Else it is rgb.
	else if (currentRenderingMode == RenderingMode::RGB_ASCII)
	{
		return MinimizeASCIIRGB(size, x, y);
	}
	else
	{
		return MinimizePIXELRGB(size, x, y);
	}
}

size_t RayTracingManager::MinimizeASCII(const size_t size, const size_t x, const size_t y)
{
	size_t newlines = 0;
	size_t addedChars = 0;

	//We hold a pointer to the spot in the buffer with the latest color.
	char* latestColor = nullptr;

	for (size_t i = 0; i < size;)
	{
		//If we are handling the end of a line.
		//If i + 1 is the end of the line. A line is 4 characters times the size of x, then +1.
		if (i != 0 && (i % (SIZE_8BIT_ASCII * x)) == 0)
		{
			++newlines;

			m_minimizedResultArray[addedChars] = '\n';
			++addedChars;

			//Stop iterating if the amount of lines equals the height.
			if (newlines == y)
			{
				break;
			}
		}

		//If its not the same color add the whole escape sequence and update latest color.
		if (
			!latestColor ||
			latestColor[0] != m_hostResultArray[i + 0] ||
			latestColor[1] != m_hostResultArray[i + 1] ||
			latestColor[2] != m_hostResultArray[i + 2]
			)
		{
			//Move the pointer to the spot in the array with the color.
			latestColor = m_hostResultArray.get() + i;

			char data[12] =
			{
				'\x1b', '[',				//Escape character
				'3', '8', ';',				//Keycode for foreground
				'5', ';',					//Keycode for foreground
				m_hostResultArray[i + 0],
				m_hostResultArray[i + 1],
				m_hostResultArray[i + 2],	//Index
				'm',
				m_hostResultArray[i + 3]	//Character data.
			};

			//Copy the escape sequence and data to the minimized result.
			memcpy(m_minimizedResultArray.get() + addedChars, data, 12);

			addedChars += 12;
		}
		//Only add the data and not the escape sequence.
		else
		{
			m_minimizedResultArray[addedChars] = m_hostResultArray[i + 3];

			addedChars += 1;
		}

		//Move 4 characters forward.
		i += SIZE_8BIT_ASCII;
	}

	return addedChars;
}

size_t RayTracingManager::MinimizePIXEL(const size_t size, const size_t x, const size_t y)
{
	size_t newlines = 0;
	size_t addedChars = 0;

	//We hold a pointer to the spot in the buffer with the latest color.
	char* latestColor = nullptr;

	for (size_t i = 0; i < size;)
	{
		//If we are handling the end of a line.
		//If i + 1 is the end of the line. A line is 3 characters times the size of x, then +1.
		if (i != 0 && (i % (SIZE_8BIT_PIXEL * x)) == 0)
		{
			++newlines;

			m_minimizedResultArray[addedChars] = '\n';
			++addedChars;

			//Stop iterating if the amount of lines equals the height.
			if (newlines == y)
			{
				break;
			}
		}

		//If its not the same color add the whole escape sequence and update latest color.
		if (
			!latestColor ||
			latestColor[0] != m_hostResultArray[i + 0] ||
			latestColor[1] != m_hostResultArray[i + 1] ||
			latestColor[2] != m_hostResultArray[i + 2]
			)
		{
			//Move the pointer to the spot in the array with the color.
			latestColor = m_hostResultArray.get() + i;

			char data[12] =
			{
				'\x1b', '[',				//Escape character
				'4', '8', ';',				//Keycode for background
				'5', ';',					//Keycode for background
				m_hostResultArray[i + 0],
				m_hostResultArray[i + 1],
				m_hostResultArray[i + 2],	//Index
				'm', ' '
			};

			//Copy the escape sequence and data to the minimized result.
			memcpy(m_minimizedResultArray.get() + addedChars, data, 12);

			addedChars += 12;
		}
		//Only add the data and not the escape sequence.
		else
		{
			m_minimizedResultArray[addedChars] = ' ';

			addedChars += 1;
		}

		//Move 3 characters forward.
		i += SIZE_8BIT_PIXEL;
	}

	return addedChars;
}

size_t RayTracingManager::MinimizeASCIIRGB(const size_t size, const size_t x, const size_t y)
{
	size_t newlines = 0;
	size_t addedChars = 0;

	//We hold a pointer to the spot in the buffer with the latest color.
	char* latestColor = nullptr;

	for (size_t i = 0; i < size;)
	{
		//If we are handling the end of a line.
		//If i is the end of the line. A line is 10 characters times the size of x, then +1.
		if (i != 0 && (i % (SIZE_RGB_ASCII * x)) == 0)
		{
			++newlines;

			m_minimizedResultArray[addedChars] = '\n';
			++addedChars;

			//Stop iterating if the amount of lines equals the height.
			if (newlines == y)
			{
				break;
			}
		}

		//If its not the same color add the whole escape sequence and update latest color.
		if (
			!latestColor ||
			latestColor[0] != m_hostResultArray[i + 0] || latestColor[1] != m_hostResultArray[i + 1] || latestColor[2] != m_hostResultArray[i + 2] ||		//R
			latestColor[3] != m_hostResultArray[i + 3] || latestColor[4] != m_hostResultArray[i + 4] || latestColor[5] != m_hostResultArray[i + 5] ||	//G
			latestColor[6] != m_hostResultArray[i + 6] || latestColor[7] != m_hostResultArray[i + 7] || latestColor[8] != m_hostResultArray[i + 8]		//B
			)
		{
			//Move the pointer to the spot in the array with the color.
			latestColor = m_hostResultArray.get() + i;

			char data[20] =
			{
				'\x1b', '[',					//Escape character
				'3', '8', ';',					//Keycode for foreground
				'2', ';',						//Keycode for foreground
				m_hostResultArray[i + 0],
				m_hostResultArray[i + 1],
				m_hostResultArray[i + 2], ';',	//R
				m_hostResultArray[i + 3],
				m_hostResultArray[i + 4],
				m_hostResultArray[i + 5], ';',	//G
				m_hostResultArray[i + 6],
				m_hostResultArray[i + 7],
				m_hostResultArray[i + 8],		//B
				'm',
				m_hostResultArray[i + 9]		//Character data.
			};

			//Copy the escape sequence and data to the minimized result.
			memcpy(m_minimizedResultArray.get() + addedChars, data, 20);

			addedChars += 20;
		}
		//Only add the data and not the escape sequence.
		else
		{
			m_minimizedResultArray[addedChars] = m_hostResultArray[i + 9];

			addedChars += 1;
		}

		//Move 10 characters forward.
		i += SIZE_RGB_ASCII;
	}

	return addedChars;
}

size_t RayTracingManager::MinimizePIXELRGB(const size_t size, const size_t x, const size_t y)
{
	size_t newlines = 0;
	size_t addedChars = 0;

	//We hold a pointer to the spot in the buffer with the latest color.
	char* latestColor = nullptr;

	for (size_t i = 0; i < size;)
	{
		//If we are handling the end of a line.
		//If i + 1 is the end of the line. A line is 9 characters times the size of x, then +1.
		if (i != 0 && (i % (SIZE_RGB_PIXEL * x)) == 0)
		{
			++newlines;

			m_minimizedResultArray[addedChars] = '\n';
			++addedChars;

			//Stop iterating if the amount of lines equals the height.
			if (newlines == y)
			{
				break;
			}
		}

		//If its not the same color add the whole escape sequence and update latest color.
		if (
			!latestColor ||
			latestColor[0] != m_hostResultArray[i + 0] || latestColor[1] != m_hostResultArray[i + 1] || latestColor[2] != m_hostResultArray[i + 2] ||		//R
			latestColor[3] != m_hostResultArray[i + 3] || latestColor[4] != m_hostResultArray[i + 4] || latestColor[5] != m_hostResultArray[i + 5] ||	//G
			latestColor[6] != m_hostResultArray[i + 6] || latestColor[7] != m_hostResultArray[i + 7] || latestColor[8] != m_hostResultArray[i + 8]		//B
			)
		{
			//Move the pointer to the spot in the array with the color.
			latestColor = m_hostResultArray.get() + i;

			char data[20] =
			{
				'\x1b', '[',					//Escape character
				'4', '8', ';',					//Keycode for background
				'2', ';',						//Keycode for background
				m_hostResultArray[i + 0],
				m_hostResultArray[i + 1],
				m_hostResultArray[i + 2], ';',	//R
				m_hostResultArray[i + 3],
				m_hostResultArray[i + 4],
				m_hostResultArray[i + 5], ';',	//G
				m_hostResultArray[i + 6],
				m_hostResultArray[i + 7],
				m_hostResultArray[i + 8],		//B
				'm',
				' '
			};

			//Copy the escape sequence and data to the minimized result.
			memcpy(m_minimizedResultArray.get() + addedChars, data, 20);

			addedChars += 20;
		}
		//Only add the data and not the escape sequence.
		else
		{
			m_minimizedResultArray[addedChars] = ' ';

			addedChars += 1;
		}

		//Move 9 characters forward.
		i += SIZE_RGB_PIXEL;
	}

	return addedChars;
}
