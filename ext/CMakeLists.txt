file(ARCHIVE_EXTRACT INPUT "${CMAKE_CURRENT_SOURCE_DIR}/src.zip" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
file(ARCHIVE_EXTRACT INPUT "${CMAKE_CURRENT_SOURCE_DIR}/vendor.zip" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")

if(NOT TARGET glfw)
  set(GLFW_INSTALL OFF)
  set(GLFW_BUILD_DOCS OFF)
  message(STATUS "[glfw]")
  add_subdirectory(src/glfw)
  add_library(glfw::glfw ALIAS glfw)
endif()

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

if(NOT TARGET klib::klib)
  message(STATUS "[klib]")
  add_subdirectory(src/klib)
endif()

if(NOT TARGET dyvk::dyvk)
  message(STATUS "[dyvk]")
  add_subdirectory(src/dyvk)
endif()

message(STATUS "[gsl]")
add_subdirectory(vendor/gsl)

message(STATUS "[vma]")
add_subdirectory(src/vma)
add_library(vma)
add_library(vma::vma ALIAS vma)
target_link_libraries(vma PUBLIC
  GPUOpen::VulkanMemoryAllocator
  dyvk::dyvk
)
target_include_directories(vma SYSTEM PUBLIC src/vma/include)
target_sources(vma PRIVATE vendor/vma_usage.cpp)
target_compile_definitions(vma PUBLIC
  VMA_STATIC_VULKAN_FUNCTIONS=0
  VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)

message(STATUS "[dear imgui]")
add_library(dear_imgui)
add_library(dear_imgui::dear_imgui ALIAS dear_imgui)

target_include_directories(dear_imgui SYSTEM PUBLIC src/dear_imgui)

target_sources(dear_imgui PRIVATE
  src/dear_imgui/imconfig.h
  src/dear_imgui/imgui_demo.cpp
  src/dear_imgui/imgui_draw.cpp
  src/dear_imgui/imgui_internal.h
  src/dear_imgui/imgui_tables.cpp
  src/dear_imgui/imgui_widgets.cpp
  src/dear_imgui/imgui.cpp
  src/dear_imgui/imgui.h

  src/dear_imgui/backends/imgui_impl_glfw.cpp
  src/dear_imgui/backends/imgui_impl_glfw.h
  src/dear_imgui/backends/imgui_impl_vulkan.cpp
  src/dear_imgui/backends/imgui_impl_vulkan.h
)

target_compile_definitions(dear_imgui PUBLIC
  GLFW_INCLUDE_VULKAN
)

target_link_libraries(dear_imgui PRIVATE
  glfw::glfw
  dyvk::dyvk
)

if(CMAKE_CXX_COMPILER_ID STREQUAL Clang OR CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  target_compile_options(dear_imgui PRIVATE
    -Wno-conversion -Wno-extra
  )
endif()

message(STATUS "[stb]")
add_subdirectory(vendor/stb)

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
