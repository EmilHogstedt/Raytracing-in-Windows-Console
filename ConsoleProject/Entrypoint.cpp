#include "pch.h"
#include "Engine3D.h"

int main()
{
    Engine3D::CreateEngine();
    Engine3D::GetInstance()->Start();
    //Always returns true. Quits and cleans up by closing the console through CEvents.
    while(Engine3D::GetInstance()->Run())
    { }
    
    Engine3D::GetInstance()->CleanUp();
}
