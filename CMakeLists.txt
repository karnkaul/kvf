cmake_minimum_required(VERSION 3.23)

project(kvf VERSION "0.4.2")

set(KVF_RESOURCE_BUFFERING 2 CACHE STRING "[Int] kvf resource buffering [2-8]")
option(KVF_USE_FREETYPE "Build and use freetype" ON)
option(KVF_BUILD_EXAMPLE "Build kvf example" ${PROJECT_IS_TOP_LEVEL})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_DEBUG_POSTFIX "-d")

set(ext_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/ext/src")

add_subdirectory(ext)

add_subdirectory(lib)

if(KVF_BUILD_EXAMPLE)
  add_subdirectory(example)
endif()
