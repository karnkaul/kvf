include(FetchContent)

if(NOT TARGET glfw)
  FetchContent_Declare(glfw
    GIT_REPOSITORY "https://github.com/glfw/glfw"
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
    SOURCE_DIR "${ext_src_dir}/glfw"
  )

  set(GLFW_VULKAN_STATIC OFF)
  set(GLFW_INSTALL OFF)
  set(GLFW_BUILD_DOCS OFF)
  set(GLFW_BUILD_EXAMPLES OFF)
  set(GLFW_BUILD_TESTS OFF)
  FetchContent_MakeAvailable(glfw)
  add_library(glfw::glfw ALIAS glfw)
endif()

add_executable(${PROJECT_NAME}-example WIN32)

target_link_libraries(${PROJECT_NAME}-example PRIVATE
  kvf::kvf
  glfw::glfw
  klib::klib-warnings
)

target_include_directories(${PROJECT_NAME}-example PRIVATE
  src
)

target_sources(${PROJECT_NAME}-example PRIVATE
  src/main.cpp
)
