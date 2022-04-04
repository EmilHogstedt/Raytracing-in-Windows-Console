#include "pch.h"
#include "Scene3D.h"

//Here the starter objects are created.
void Scene3D::Init()
{
	gpuErrchk(cudaMalloc(&(m_deviceObjects.m_deviceArray1), FIVE_MEGABYTES));
	gpuErrchk(cudaMemset(m_deviceObjects.m_deviceArray1, 0, FIVE_MEGABYTES));
	m_deviceObjects.allocatedBytes = FIVE_MEGABYTES;
	m_deviceObjects.count = 0;
	m_deviceObjects.using1st = true;

	gpuErrchk(cudaMalloc(&(m_devicePlanes.m_deviceArray1), FIVE_MEGABYTES));
	gpuErrchk(cudaMemset(m_devicePlanes.m_deviceArray1, 0, FIVE_MEGABYTES));
	m_devicePlanes.allocatedBytes = FIVE_MEGABYTES;
	m_devicePlanes.count = 0;
	m_devicePlanes.using1st = true;

	gpuErrchk(cudaMalloc(&(m_deviceSpheres.m_deviceArray1), FIVE_MEGABYTES));
	gpuErrchk(cudaMemset(m_deviceSpheres.m_deviceArray1, 0, FIVE_MEGABYTES));
	m_deviceSpheres.allocatedBytes = FIVE_MEGABYTES;
	m_deviceSpheres.count = 0;
	m_deviceSpheres.using1st = true;

	//Temporary.
	CreateSphere(7.0f, Vector3(0.0f, 10.0f, 20.0f), Vector3(255.0f, 1.0f, 1.0f));
	CreateSphere(6.0f, Vector3(5.0f, 10.0f, 20.0f), Vector3(1.0f, 255.0f, 1.0f));
	CreateSphere(10.0f, Vector3(10.0f, 10.0f, 40.0f), Vector3(1.0f, 1.0f, 255.0f));
	CreateSphere(3.0f, Vector3(5.0f, 10.0f, 20.0f), Vector3(225.0f, 210.0f, 20.0f));
	CreateSphere(4.0f, Vector3(-5.0f, 10.0f, 40.0f), Vector3(225.0f, 10.0f, 220.0f));
	CreatePlane(Vector3(0.0f, -3.0f, 0.0f), Vector3(0.0f, 1.0f, 0.0f), Vector3(100.0f, 100.0f, 100.0f));
}

void Scene3D::CreateSphere(float radius, Vector3 middlePos, Vector3 color)
{
	//See so that we have space for the object pointers on the GPU.
	if (m_deviceObjects.allocatedBytes < (m_deviceObjects.count + 1) * sizeof(Object3D*))
	{
		if (m_deviceObjects.allocatedBytes >= HUNDRED_MEGABYTES)
		{
			throw std::runtime_error("Error! Out of dedicated memory when trying to create an object.");
		}
		//If we do not we allocate more memory and copy over the current array to the new memory.
		//This is done elegantly by changing the current array that is used.
		m_deviceObjects.using1st = !m_devicePlanes.using1st;
		Object3D** newArray;
		Object3D** oldArray;
		if (m_deviceObjects.using1st)
		{
			newArray = m_deviceObjects.m_deviceArray1;
			oldArray = m_deviceObjects.m_deviceArray2;
		}
		else
		{
			newArray = m_deviceObjects.m_deviceArray2;
			oldArray = m_deviceObjects.m_deviceArray1;
		}
		gpuErrchk(cudaMalloc(&newArray, m_deviceObjects.allocatedBytes * 2));
		gpuErrchk(cudaMemset(newArray, 0, m_deviceObjects.allocatedBytes * 2));
		gpuErrchk(cudaMemcpy(newArray, oldArray, m_deviceObjects.allocatedBytes, cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaFree(oldArray));
		m_deviceObjects.allocatedBytes *= 2;
	}

	//Check which we are currently using.
	Object3D** currentObjectArray;
	if (m_deviceObjects.using1st)
	{
		currentObjectArray = m_deviceObjects.m_deviceArray1;
	}
	else
	{
		currentObjectArray = m_deviceObjects.m_deviceArray2;
	}

	//Now we can check if we have enough memory for the new subtype of Object3D.
	//The process is the same as above.
	if (m_deviceSpheres.allocatedBytes < (m_deviceSpheres.count + 1) * sizeof(Sphere))
	{
		if (m_deviceSpheres.allocatedBytes >= HUNDRED_MEGABYTES)
		{
			throw std::runtime_error("Error! Out of dedicated memory when trying to create an object.");
		}
		m_deviceSpheres.using1st = !m_deviceSpheres.using1st;
		Sphere* newArray;
		Sphere* oldArray;
		if (m_deviceSpheres.using1st)
		{
			newArray = m_deviceSpheres.m_deviceArray1;
			oldArray = m_deviceSpheres.m_deviceArray2;
		}
		else
		{
			newArray = m_deviceSpheres.m_deviceArray2;
			oldArray = m_deviceSpheres.m_deviceArray1;
		}
		gpuErrchk(cudaMalloc(&newArray, m_deviceSpheres.allocatedBytes * 2));
		gpuErrchk(cudaMemset(newArray, 0, m_deviceSpheres.allocatedBytes * 2));
		gpuErrchk(cudaMemcpy(newArray, oldArray, m_deviceSpheres.allocatedBytes, cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaFree(oldArray));
		m_deviceSpheres.allocatedBytes *= 2;
	}

	//Copy the new object to the correct spot in GPU memory in the correct array.
	Sphere* currentArray;
	if (m_deviceSpheres.using1st)
	{
		currentArray = m_deviceSpheres.m_deviceArray1;
	}
	else
	{
		currentArray = m_deviceSpheres.m_deviceArray2;
	}

	Sphere newObject = Sphere(middlePos, radius, color);
	gpuErrchk(cudaMemcpy(currentArray + m_deviceSpheres.count, &newObject, sizeof(Sphere), cudaMemcpyHostToDevice));
	
	//Now we have to copy the pointer of the object we just added to the objectPointer array on the GPU.
	Object3D* temp[1];
	temp[0] = currentArray + m_deviceSpheres.count;
	gpuErrchk(cudaMemcpy(currentObjectArray + m_deviceObjects.count, temp, sizeof(Object3D*), cudaMemcpyHostToDevice));
	
	m_deviceSpheres.count++;
	m_deviceObjects.count++;
}

void Scene3D::CreatePlane(Vector3 middlePos, Vector3 normal, Vector3 color)
{
	//See so that we have space for the object pointers on the GPU.
	if (m_deviceObjects.allocatedBytes < (m_deviceObjects.count + 1) * sizeof(Object3D*))
	{
		if (m_deviceObjects.allocatedBytes >= HUNDRED_MEGABYTES)
		{
			throw std::runtime_error("Error! Out of dedicated memory when trying to create an object.");
		}
		//If we do not we allocate more memory and copy over the current array to the new memory.
		//This is done elegantly by changing the current array that is used.
		m_deviceObjects.using1st = !m_devicePlanes.using1st;
		Object3D** newArray;
		Object3D** oldArray;
		if (m_deviceObjects.using1st)
		{
			newArray = m_deviceObjects.m_deviceArray1;
			oldArray = m_deviceObjects.m_deviceArray2;
		}
		else
		{
			newArray = m_deviceObjects.m_deviceArray2;
			oldArray = m_deviceObjects.m_deviceArray1;
		}
		gpuErrchk(cudaMalloc(&newArray, m_deviceObjects.allocatedBytes * 2));
		gpuErrchk(cudaMemset(newArray, 0, m_deviceObjects.allocatedBytes * 2));
		gpuErrchk(cudaMemcpy(newArray, oldArray, m_deviceObjects.allocatedBytes, cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaFree(oldArray));
		m_deviceObjects.allocatedBytes *= 2;
	}

	//Check which we are currently using.
	Object3D** currentObjectArray;
	if (m_deviceObjects.using1st)
	{
		currentObjectArray = m_deviceObjects.m_deviceArray1;
	}
	else
	{
		currentObjectArray = m_deviceObjects.m_deviceArray2;
	}

	//Now we can check if we have enough memory for the new subtype of Object3D.
	//The process is the same as above.
	if (m_devicePlanes.allocatedBytes < (m_devicePlanes.count + 1) * sizeof(Plane))
	{
		if (m_devicePlanes.allocatedBytes >= HUNDRED_MEGABYTES)
		{
			throw std::runtime_error("Error! Out of dedicated memory when trying to create an object.");
		}
		m_devicePlanes.using1st = !m_devicePlanes.using1st;
		Plane* newArray;
		Plane* oldArray;
		if (m_devicePlanes.using1st)
		{
			newArray = m_devicePlanes.m_deviceArray1;
			oldArray = m_devicePlanes.m_deviceArray2;
		}
		else
		{
			newArray = m_devicePlanes.m_deviceArray2;
			oldArray = m_devicePlanes.m_deviceArray1;
		}
		gpuErrchk(cudaMalloc(&newArray, m_devicePlanes.allocatedBytes * 2));
		gpuErrchk(cudaMemset(newArray, 0, m_devicePlanes.allocatedBytes * 2));
		gpuErrchk(cudaMemcpy(newArray, oldArray, m_devicePlanes.allocatedBytes, cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaFree(oldArray));
		m_devicePlanes.allocatedBytes *= 2;
	}

	//Copy the new object to the correct spot in GPU memory in the correct array.
	Plane* currentArray;
	if (m_devicePlanes.using1st)
	{
		currentArray = m_devicePlanes.m_deviceArray1;
	}
	else
	{
		currentArray = m_devicePlanes.m_deviceArray2;
	}

	Plane newObject = Plane(middlePos, normal, color);
	gpuErrchk(cudaMemcpy(currentArray + m_devicePlanes.count, &newObject, sizeof(Plane), cudaMemcpyHostToDevice));
	
	Object3D* temp[1];
	temp[0] = currentArray + m_devicePlanes.count;
	//Now we have to copy the pointer of the object we just added to the objectPointer array on the GPU.
	gpuErrchk(cudaMemcpy(currentObjectArray + m_deviceObjects.count, temp, sizeof(Object3D*), cudaMemcpyHostToDevice));

	m_devicePlanes.count++;
	m_deviceObjects.count++;
}

//Update all objects
void Scene3D::Update(long double deltaTime)
{
	//Let GPU update objects instead
}

void Scene3D::CleanUp()
{
	if (m_deviceObjects.using1st)
	{
		gpuErrchk(cudaFree(m_deviceObjects.m_deviceArray1));
	}
	else
	{
		gpuErrchk(cudaFree(m_deviceObjects.m_deviceArray2));
	}

	if (m_devicePlanes.using1st)
	{
		gpuErrchk(cudaFree(m_devicePlanes.m_deviceArray1));
	}
	else
	{
		gpuErrchk(cudaFree(m_devicePlanes.m_deviceArray2));
	}

	if (m_deviceSpheres.using1st)
	{
		gpuErrchk(cudaFree(m_deviceSpheres.m_deviceArray1));
	}
	else
	{
		gpuErrchk(cudaFree(m_deviceSpheres.m_deviceArray2));
	}
}

//Used to send the devicepointer to the objects to the raytracer.
DeviceObjectArray<Object3D*> Scene3D::GetObjects()
{
	return m_deviceObjects;
}