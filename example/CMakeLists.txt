include(FetchContent)

add_executable(${PROJECT_NAME}-example)

target_link_libraries(${PROJECT_NAME}-example PRIVATE
  kvf::kvf
  klib::klib-warnings
)

target_include_directories(${PROJECT_NAME}-example PRIVATE
  src
)

target_sources(${PROJECT_NAME}-example PRIVATE
  src/main.cpp
)
