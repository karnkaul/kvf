add_library(${PROJECT_NAME})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23)

set(freetype_val 0)

if(KVF_USE_FREETYPE)
  set(freetype_val 1)
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC
  GLFW_INCLUDE_VULKAN
  KVF_RESOURCE_BUFFERING=${KVF_RESOURCE_BUFFERING}
  KVF_USE_FREETYPE=${freetype_val}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  ms::gsl
  glm::glm
  klib::klib
  glfw::glfw
  vma::vma
  dear_imgui::dear_imgui
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  stb-image::stb-image
  $<$<BOOL:${KVF_USE_FREETYPE}>:freetype::freetype>
)

target_include_directories(${PROJECT_NAME} PRIVATE
  src
)

file(GLOB_RECURSE headers LIST_DIRECTORIES false "include/*.hpp")
target_sources(${PROJECT_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS include FILES
  ${headers}
)

configure_file(cmake_src/build_version.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/build_version.hpp")
target_sources(${PROJECT_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/include" FILES
  "${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/build_version.hpp"
)

target_sources(${PROJECT_NAME} PRIVATE
  src/kvf.cpp
  src/log.hpp
  src/ttf.cpp
)
