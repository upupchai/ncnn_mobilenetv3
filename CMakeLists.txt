cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(Class_ncnn)

# compile
add_compile_options(-g)
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g")
set(CMAKE_CXX_STANDARD 11)


#option(GPU "Build the project using macro" ON)
option(NCNN_VULKAN "NCNN_VULKAN" OFF)
option(GPU_PC "GPU_PC" OFF)

set(Third_party_dir ${CMAKE_SOURCE_DIR}/3rdparty)

if(GPU_PC)
#    add_definitions(-D DEBUG_X86_64)
    set(NCNN_DIR ${Third_party_dir}/install_newcpu)
    include_directories(${Third_party_dir}/1.2.176.1/x86_64/include)
    link_libraries(${Third_party_dir}/1.2.176.1/x86_64/lib)
else()
    set(NCNN_DIR ${Third_party_dir}/install_cpu)

endif()
if(NCNN_VULKAN)
    add_definitions(-D NCNN_VULKAN1)
endif()

find_package(CUDA)

include_directories(${NCNN_DIR}/include)
link_directories(${NCNN_DIR}/lib)



set(OpenCV_DIR  ${Third_party_dir}/opencv_pc/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(${PROJECT_NAME} mobilenetv3.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ncnn)

#target_link_libraries(${PROJECT_NAME} ${Glslang_LIBS}
#        libncnn.a libglslang.a libGenericCodeGen.a  libOGLCompiler.a libOSDependent.a libSPIRV.a vulkan glslang)

#PRIVATE ncnn glslang
#target_compile_features(${PROJECT_NAME} PRIVATE cxx_range_for)



