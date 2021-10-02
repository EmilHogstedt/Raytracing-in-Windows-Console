#include "pch.h"
#include "Engine.h"

int main()
{
    Engine::CreateEngine();
    Engine::GetInstance()->Start();
    //Always returns true. Quits and cleans up by closing the console through CEvents.
    while(Engine::GetInstance()->Run())
    { }
    
    Engine::GetInstance()->CleanUp();
}
