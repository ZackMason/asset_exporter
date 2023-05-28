set PhysXSDK=C:\Users\zack\Documents\GitHub\PhysX5\PhysX\physx
set PhysXCompiler=%PhysXSDK%\bin\win.x86_64.vc142.md
set PhysXOpt=checked

xcopy %PhysXCompiler%\%PhysXOpt%\*.dll build\bin\ /yq

cmake --build build --config Release -j 12