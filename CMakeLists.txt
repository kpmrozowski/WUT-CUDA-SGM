cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_COMPILER "/bin/gcc-10")
set(CMAKE_CXX_COMPILER "/bin/g++-10")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.6")
option(ENABLE_SAMPLE "Build the sample" ON)
set(DEBUG ON)


# add_compile_options("$<$<CONFIG:DEBUG>:--cuda-gpu-arch=sm_61>")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")
if(DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
  # add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")
endif()

set(CUDA_ARCH "-arch=sm_61" CACHE STRING "Value of the NVCC -arch option.")
if(${CMAKE_VERSION} VERSION_EQUAL 3.7 OR ${CMAKE_VERSION} VERSION_GREATER 3.7)
  option(AUTO_DETECT_ARCH "Detect local GPU compute arch automatically" ON)
endif()

project(kmroSGM VERSION 0.1.0)

add_subdirectory(src)

if(ENABLE_SAMPLE)
    add_subdirectory(sample)
endif()
