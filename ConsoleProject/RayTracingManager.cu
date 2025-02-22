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
	if (currentRenderingMode == RenderingMode::BIT_ASCII || currentRenderingMode == RenderingMode::BIT_PIXEL)
	{
		return Minimize8bit(size, x, y);
	}
	//Else it is rgb.
	else
	{
		return MinimizeRGB(size, x, y);
	}
}

size_t RayTracingManager::Minimize8bit(const size_t size, const size_t x, const size_t y)
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
				memcpy(m_minimizedResultArray.get() + addedChars, m_hostResultArray.get() + i, SIZE_8BIT);

				addedChars += SIZE_8BIT;
			}
			//Only add the data and not the escape sequence.
			else
			{
				m_minimizedResultArray[addedChars] = m_hostResultArray[i + SIZE_8BIT - 1];

				addedChars += 1;
			}

			//Move 12 characters forward.
			i += SIZE_8BIT;
		}
		//If we are handling the end of a line.
		//If i + 1 is the end of the line. A line is 12 characters times the size of x, then +1.
		else if (((i + 1) % (SIZE_8BIT * x)) == 0)
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

size_t RayTracingManager::MinimizeRGB(const size_t size, const size_t x, const size_t y)
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
				memcpy(m_minimizedResultArray.get() + addedChars, m_hostResultArray.get() + i, SIZE_RGB);

				addedChars += SIZE_RGB;
			}
			//Only add the data and not the escape sequence.
			else
			{
				m_minimizedResultArray[addedChars] = m_hostResultArray[i + SIZE_RGB - 1];

				addedChars += 1;
			}

			//Move 20 characters forward.
			i += SIZE_RGB;
		}
		//If we are handling the end of a line.
		//If i + 1 is the end of the line. A line is 20 characters times the size of x, then +1.
		else if (((i + 1) % (SIZE_RGB * x)) == 0)
		{
			++newlines;

			m_minimizedResultArray[addedChars] = '\n';
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
