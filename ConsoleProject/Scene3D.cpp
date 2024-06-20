#include "pch.h"
#include "Scene3D.h"
#include "PrintMachine.h"

void Scene3D::Init()
{
	//Allocate device memory for pointers to the objects.
	gpuErrchk(cudaMalloc(&(m_deviceObjects.m_deviceArray), 80));// FIVE_MEGABYTES));
	gpuErrchk(cudaMemset(m_deviceObjects.m_deviceArray, 0, 80));//FIVE_MEGABYTES));
	m_deviceObjects.allocatedBytes = 80;//FIVE_MEGABYTES;
	m_deviceObjects.count = 0;

	//Allocate device memory for the object data.
	//MAKE SURE ENOUGH MEMORY IS ALLOCATED FROM THE START!!
	gpuErrchk(cudaMalloc(&(m_devicePlanes.m_deviceArray), FIVE_MEGABYTES));
	gpuErrchk(cudaMemset(m_devicePlanes.m_deviceArray, 0, FIVE_MEGABYTES));
	m_devicePlanes.allocatedBytes = FIVE_MEGABYTES;
	m_devicePlanes.count = 0;

	gpuErrchk(cudaMalloc(&(m_deviceSpheres.m_deviceArray), FIVE_MEGABYTES));
	gpuErrchk(cudaMemset(m_deviceSpheres.m_deviceArray, 0, FIVE_MEGABYTES));
	m_deviceSpheres.allocatedBytes = FIVE_MEGABYTES;
	m_deviceSpheres.count = 0;

	//Temporary.
	CreateSphere(7.0f, MyMath::Vector3(0.0f, 10.0f, 20.0f), MyMath::Vector3(255.0f, 1.0f, 1.0f));
	CreateSphere(6.0f, MyMath::Vector3(5.0f, 10.0f, 20.0f), MyMath::Vector3(1.0f, 255.0f, 1.0f));
	CreateSphere(10.0f, MyMath::Vector3(10.0f, 10.0f, 40.0f), MyMath::Vector3(1.0f, 1.0f, 255.0f));
	CreateSphere(3.0f, MyMath::Vector3(5.0f, 10.0f, 20.0f), MyMath::Vector3(225.0f, 210.0f, 20.0f));
	CreateSphere(4.0f, MyMath::Vector3(-5.0f, 10.0f, 40.0f), MyMath::Vector3(225.0f, 10.0f, 220.0f));
	CreatePlane(MyMath::Vector3(0.0f, -3.0f, 0.0f), MyMath::Vector3(0.0f, 1.0f, 0.0f), MyMath::Vector3(100.0f, 100.0f, 100.0f));
}

void Scene3D::CreateSphere(const float radius, const MyMath::Vector3& middlePos, const MyMath::Vector3& color)
{
	//Make sure that we have space for the object pointers on the GPU.
	CheckDeviceObjectsPtrMemory();

	//Now we can check if we have enough memory for the new subtype of Object3D.
	if (!CheckDeviceObjectsDataMemory<Sphere>(m_deviceSpheres))
	{
		return;
	}

	Sphere newObject = Sphere(middlePos, radius, color);

	void DEVICE_MEMORY_PTR ObjectMemoryLocation = m_deviceSpheres.m_deviceArray + m_deviceSpheres.count;
	void DEVICE_MEMORY_PTR PointerMemoryLocation = m_deviceObjects.m_deviceArray + m_deviceObjects.count;

	//Copy the new object to the correct spot in GPU memory.
	gpuErrchk(cudaMemcpy(ObjectMemoryLocation, &newObject, sizeof(Sphere), cudaMemcpyHostToDevice));

	//Copy the address of the object memory to the pointer memory.
	gpuErrchk(cudaMemcpy(PointerMemoryLocation, &ObjectMemoryLocation, sizeof(Object3D*), cudaMemcpyHostToDevice));
	
	m_deviceSpheres.count++;
	m_deviceObjects.count++;
}

void Scene3D::CreatePlane(const MyMath::Vector3& middlePos, const MyMath::Vector3& normal, const MyMath::Vector3& color)
{
	//See so that we have space for the object pointers on the GPU.
	CheckDeviceObjectsPtrMemory();

	//Now we can check if we have enough memory for the new subtype of Object3D.
	if (!CheckDeviceObjectsDataMemory<Plane>(m_devicePlanes))
	{
		return;
	}

	Plane newObject = Plane(middlePos, normal, color);

	void DEVICE_MEMORY_PTR ObjectMemoryLocation = m_devicePlanes.m_deviceArray + m_devicePlanes.count;
	void DEVICE_MEMORY_PTR PointerMemoryLocation = m_deviceObjects.m_deviceArray + m_deviceObjects.count;

	//Copy the new object to the correct spot in GPU memory.
	gpuErrchk(cudaMemcpy(ObjectMemoryLocation, &newObject, sizeof(Plane), cudaMemcpyHostToDevice));
	
	//Copy the address of the object memory to the pointer memory.
	gpuErrchk(cudaMemcpy(PointerMemoryLocation, &ObjectMemoryLocation, sizeof(Object3D*), cudaMemcpyHostToDevice));

	m_devicePlanes.count++;
	m_deviceObjects.count++;
}

//Update all objects
void Scene3D::Update(const long double deltaTime)
{
	//Let GPU update objects instead
}

void Scene3D::CleanUp()
{
	gpuErrchk(cudaFree(m_deviceObjects.m_deviceArray));
	gpuErrchk(cudaFree(m_deviceSpheres.m_deviceArray));
	gpuErrchk(cudaFree(m_devicePlanes.m_deviceArray));
}

//Used to send the devicepointer to the objects to the raytracer.
DeviceObjectArray<Object3D*> Scene3D::GetObjects()
{
	return m_deviceObjects;
}

void Scene3D::CheckDeviceObjectsPtrMemory()
{
	unsigned int nextSize = (m_deviceObjects.count + 1) * sizeof(Object3D*); //For debugging.
	if (m_deviceObjects.allocatedBytes < nextSize)
	{
		if (m_deviceObjects.allocatedBytes * 2 >= HUNDRED_MEGABYTES)
		{
			throw std::runtime_error("Error! Out of dedicated memory when trying to create an object.");
		}

		//If we do not have enough memory we allocate more memory and copy over the current array to the new memory.
		//Object3D** newArray;
		Object3D** oldArray = m_deviceObjects.m_deviceArray;

		gpuErrchk(cudaMalloc(&m_deviceObjects.m_deviceArray, m_deviceObjects.allocatedBytes * 2));
		gpuErrchk(cudaMemset(m_deviceObjects.m_deviceArray, 0, m_deviceObjects.allocatedBytes * 2));
		gpuErrchk(cudaMemcpy(m_deviceObjects.m_deviceArray, oldArray, m_deviceObjects.allocatedBytes, cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaFree(oldArray));

		//m_deviceObjects.m_deviceArray = newArray;
		m_deviceObjects.allocatedBytes *= 2;
	}
}

template<typename T>
bool Scene3D::CheckDeviceObjectsDataMemory(DeviceObjectArray<T>& deviceObjects)
{
	unsigned int nextSize = (deviceObjects.count + 1) * sizeof(T); //For debugging.
	if (deviceObjects.allocatedBytes < nextSize)
	{
		//REMADE TO THROW ERROR IF THERE IS NOT ENOUGH MEMORY. MAKE SURE TO ALLOCATE ENOUGH IN THE BEGINNING.
		//STRANGE STUFF HAPPENS IF OBJECTS ARE REALLOCATED, AS THE OLD POINTERS IN DEVICEOBJECTS ARE NO LONGER VALID!
		//#todo: Maybe revisit this in the future.
		return false;
		//throw std::runtime_error("Error! Out of dedicated memory when trying to create an object.");

		/*
		if (deviceObjects.allocatedBytes >= HUNDRED_MEGABYTES)
		{
			throw std::runtime_error("Error! Out of dedicated memory when trying to create an object.");
		}

		//If we do not have enough memory we allocate more memory and copy over the current array to the new memory.
		//T* newArray;
		T* oldArray = deviceObjects.m_deviceArray;

		gpuErrchk(cudaMalloc(&deviceObjects.m_deviceArray, deviceObjects.allocatedBytes * 2));
		gpuErrchk(cudaMemset(deviceObjects.m_deviceArray, 0, deviceObjects.allocatedBytes * 2));
		gpuErrchk(cudaMemcpy(deviceObjects.m_deviceArray, oldArray, deviceObjects.allocatedBytes, cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaFree(oldArray));

		//deviceObjects.m_deviceArray = newArray;
		deviceObjects.allocatedBytes *= 2;
		*/
	}

	return true;
}
