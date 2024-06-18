#include "pch.h"
#include "Engine3D.h"

int main()
{
    std::unique_ptr<Engine3D> engine = std::make_unique<Engine3D>();
    engine->Start();

    while(engine->Run())
    { }
    
    engine->CleanUp();
}
