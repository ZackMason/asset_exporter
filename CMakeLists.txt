cmake_minimum_required(VERSION 3.8.12)

project(game C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()
if ( MSVC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W4 /WX /wd4702 /wd4100 /wd4201")
endif()

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

include_directories(source)
include_directories(include)

add_executable(exporter ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(exporter ${CONAN_LIBS})

target_compile_definitions(exporter PUBLIC GAME_ASSETS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/")

set(PHYSX_ROOT_WIN
    "C:/Users/crazy/Downloads/PhysX-brickadia-4.1.2/PhysX-brickadia-4.1.2/"
)

# https://github.com/nyers33/minimal_glfw_physx/blob/main/src/CMakeLists.txt
set(PHSYX_LIBS
"PhysX_64.lib"
"PhysXCommon_64.lib"
"PhysXCooking_64.lib"
"PhysXFoundation_64.lib"

"PhysXPvdSDK_static_64.lib"
"PhysXVehicle_static_64.lib"
"PhysXExtensions_static_64.lib"
"PhysXCharacterKinematic_static_64.lib"
)

set(PHSYX_LIBS_ROOT_DIR "${PHYSX_ROOT_WIN}/physx/bin/win.x86_64.vc143.md")

include_directories("${PHYSX_ROOT_WIN}/physx/include" )
include_directories("${PHYSX_ROOT_WIN}/pxshared/include" )

foreach(lib ${PHSYX_LIBS})
    target_link_libraries(exporter optimized "${PHSYX_LIBS_ROOT_DIR}/checked/${lib}")
    target_link_libraries(exporter debug "${PHSYX_LIBS_ROOT_DIR}/debug/${lib}")
endforeach()
