#include "pch.h"
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
		resultArray[row * x + column] = '\n';
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
	
	//Write data to the deviceBackbuffer.
	resultArray[row * x + column] = data;
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
	cudaMemcpy(printMachine->GetBackBuffer(), deviceResultArray, (printMachine->GetWidth() + 1) * printMachine->GetHeight(), cudaMemcpyDeviceToHost);
	memset(printMachine->GetBackBufferSwap(), 1, sizeof(int));
	backBufferMutex->unlock();
	return;
}