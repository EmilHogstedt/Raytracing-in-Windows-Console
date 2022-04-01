#include "pch.h"
#include "RayTracing.cuh"

__global__ void UpdateObjects(
	Object3D** objects,
	unsigned int count,
	double dt
)
{
	if (blockIdx.x * blockDim.x + threadIdx.x >= count)
	{
		return;
	}
	
	Object3D* object = objects[blockIdx.x * blockDim.x + threadIdx.x];
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
	if (blockIdx.x * blockDim.x + threadIdx.x >= x || blockIdx.y * blockDim.y + threadIdx.y >= y)
	{
		return;
	}
	if (blockIdx.x * blockDim.x + threadIdx.x == (x - 1))
	{
		resultArray[blockIdx.y * x * blockDim.y + threadIdx.y * x + blockIdx.x * blockDim.x + threadIdx.x] = '\n';
		return;
	}
	float convertedY = ((float)y - (float)(blockIdx.y * blockDim.y + threadIdx.y) * 2.0f) / (float)y;
	float convertedX = 2.0f * (((float)(blockIdx.x * blockDim.x + threadIdx.x) - ((float)x * 0.5f)) / (float)x);
	
	Vector4 pixelVSpace = Vector4(convertedX * element1, convertedY * element2, 1.0f, 0.0f);
	Vector4 tempDirectionWSpace = params->inverseVMatrix.Mult(pixelVSpace);
	Vector3 directionWSpace = Vector3(tempDirectionWSpace.x, tempDirectionWSpace.y, tempDirectionWSpace.z);
	directionWSpace = directionWSpace.Normalize();

	char data = ' ';
	float closest = 99999999.f;
	float shadingValue = 0.0f;
	
	Vector3 cameraPos = params->camPos;
	
	unsigned int localCount = count;
	for (size_t i = 0; i < count; i++)
	{
		//Ray-Sphere intersection test.
		ObjectType type = (objects[i])->GetType();
		if (type == SphereType)
		{
			Sphere localSphere = *((Sphere*)(objects[i]));
			Vector3 objectToCam = cameraPos - localSphere.GetPos();
			float radius = localSphere.GetRadius();

			float a = Dot(directionWSpace, directionWSpace);
			float b = 2.0f * Dot(directionWSpace, objectToCam);
			float c = Dot(objectToCam, objectToCam) - (radius * radius);

			float discriminant = b * b - 4.0f * a * c;

			//It hit
			if (discriminant >= 0.0f)
			{
				float t1 = (-b + sqrt(discriminant)) / (2.0f * a);
				float t2 = (-b - sqrt(discriminant)) / (2.0f * a);

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
					shadingValue = Dot(normalSphere, Vector3() - (directionWSpace.Normalize()));
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
						Vector3 p = Vector3(cameraPos.x + directionWSpace.x * t1, cameraPos.y + directionWSpace.y * t1, cameraPos.z + directionWSpace.z * t1); //Overwrite * and + operator.
						if (p.x > -7.0f && p.x < 7.0f && p.z > 12.0f && p.z < 35.0f)
						{
							closest = t1;
							shadingValue = Dot(planeNormal, Vector3() - directionWSpace);
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
	//printf("%s\n", data);
	//Myblock-y * width of whole screen * height of 1 block = the y-block we are on in terms of elements in a 1D array.
	//my threadIdx.y within the correctblock * width of whole screen.
	//my blockid.x * width of one block + this thread id.
	//These 3 added gives the index within the 1D array.
	resultArray[blockIdx.y * x * blockDim.y + threadIdx.y * x + blockIdx.x * blockDim.x + threadIdx.x] = data;
	
	return;
	
}

void RayTracingWrapper(size_t x, size_t y, float element1, float element2, DeviceObjectArray<Object3D*> deviceObjects, RayTracingParameters* deviceParams, char* deviceResultArray, double dt, char* hostResultArray)
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
	
	UpdateObjects<<<gridDims, blockDims>>>(
		deviceObjects.using1st ? deviceObjects.m_deviceArray1 : deviceObjects.m_deviceArray2,
		deviceObjects.count,
		dt
	);
	
	//Do the raytracing. Calculate x and y dimensions in blocks depending on screensize.
	//1 thread per pixel.
	gridDims.x = std::ceil((x + 1) / 16.0);
	gridDims.y = std::ceil(y / 16.0);
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
	gpuErrchk(cudaDeviceSynchronize());
	return;
}