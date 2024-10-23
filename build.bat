set PhysXSDK=C:\Users\crazy\Documents\GitHub\PhysX\physx
set PhysXCompiler=%PhysXSDK%\bin\win.x86_64.vc143.md
set PhysXOpt=release

xcopy %PhysXCompiler%\%PhysXOpt%\*.dll build\bin\ /yq

cmake --build build --config Release -j 12
@REM cmake --build debug --config Debug -j 12