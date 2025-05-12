file(ARCHIVE_EXTRACT INPUT "${CMAKE_CURRENT_SOURCE_DIR}/src.zip" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")

if(NOT TARGET klib)
  message(STATUS "[klib]")
  add_subdirectory(src/klib)
endif()

message(STATUS "[gsl]")
add_subdirectory(src/gsl)

message(STATUS "[dyvk]")
add_subdirectory(src/dyvk)

message(STATUS "[vma]")
add_subdirectory(src/vma)
add_library(vma)
add_library(vma::vma ALIAS vma)
target_link_libraries(vma PUBLIC
  GPUOpen::VulkanMemoryAllocator
  dyvk::dyvk
)
target_include_directories(vma SYSTEM PUBLIC src/vma/include)
target_sources(vma PRIVATE src/vma/src/vk_mem_alloc.cpp)
target_compile_definitions(vma PUBLIC
  VMA_STATIC_VULKAN_FUNCTIONS=0
  VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)

set(GLFW_INSTALL OFF)
set(GLFW_BUILD_DOCS OFF)
message(STATUS "[glfw]")
add_subdirectory(src/glfw)
add_library(glfw::glfw ALIAS glfw)

message(STATUS "[dear imgui]")
add_subdirectory(src/dear_imgui)

message(STATUS "[stb]")
add_subdirectory(src/stb)

if(NOT TARGET glm)
  set(GLM_ENABLE_CXX_20 ON)
  message(STATUS "[glm]")
  add_subdirectory(src/glm)
  target_compile_definitions(glm PUBLIC
    GLM_FORCE_XYZW_ONLY
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_SILENT_WARNINGS
    GLM_ENABLE_EXPERIMENTAL
    GLM_EXT_INCLUDED
  )
endif()

if(KVF_USE_FREETYPE)
  if(NOT TARGET freetype)
    set(FT_DISABLE_ZLIB ON)
    set(FT_DISABLE_BZIP2 ON)
    set(FT_DISABLE_PNG ON)
    set(FT_DISABLE_HARFBUZZ ON)
    set(FT_DISABLE_BROTLI ON)
    message(STATUS "[freetype]")
    add_subdirectory(src/freetype)
  endif()

  if(NOT TARGET freetype::freetype)
    add_library(freetype::freetype ALIAS freetype)
  endif()
endif()
