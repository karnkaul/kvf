if(NOT TARGET klib)
  add_subdirectory(src/klib)
endif()

# file(ARCHIVE_EXTRACT INPUT "${CMAKE_CURRENT_SOURCE_DIR}/src.zip" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(src/dyvk)

add_subdirectory(src/vma)
add_library(vma)
add_library(vma::vma ALIAS vma)
target_link_libraries(vma PUBLIC
  GPUOpen::VulkanMemoryAllocator
  dyvk::dyvk
)

# target_include_directories(vma SYSTEM PUBLIC src/vma/include)
target_sources(vma PRIVATE src/vma/src/vk_mem_alloc.cpp)
target_compile_definitions(vma PUBLIC
  VMA_STATIC_VULKAN_FUNCTIONS=0
  VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)

add_subdirectory(src/gsl)
