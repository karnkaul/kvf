if(NOT TARGET klib)
  add_subdirectory(src/klib)
endif()

file(ARCHIVE_EXTRACT INPUT "${CMAKE_CURRENT_SOURCE_DIR}/src.zip" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(src/gsl)

add_subdirectory(src/dyvk)

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

set(GLFW_VULKAN_STATIC OFF)
set(GLFW_INSTALL OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
add_subdirectory(src/glfw)
add_library(glfw::glfw ALIAS glfw)

add_subdirectory(src/dear_imgui)
