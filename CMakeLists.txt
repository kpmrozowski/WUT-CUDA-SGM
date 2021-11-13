cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_ARCH "-arch=sm_61" CACHE STRING "Value of the NVCC -arch option.")
if(${CMAKE_VERSION} VERSION_EQUAL 3.7 OR ${CMAKE_VERSION} VERSION_GREATER 3.7)
  option(AUTO_DETECT_ARCH "Detect local GPU compute arch automatically" ON)
endif()

option(ENABLE_SAMPLE       "Build the sample" ON)

project(kmroSGM VERSION 0.1.0)

configure_file(${PROJECT_SOURCE_DIR}/include/kmrosgm_config.h.in
               ${PROJECT_SOURCE_DIR}/include/kmrosgm_config.h
)

add_subdirectory(src)

if(ENABLE_SAMPLE)
    add_subdirectory(sample)
endif()
