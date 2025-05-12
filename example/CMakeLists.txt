include(FetchContent)

add_executable(${PROJECT_NAME}-example)

target_link_libraries(${PROJECT_NAME}-example PRIVATE
  kvf::kvf
)

target_include_directories(${PROJECT_NAME}-example PRIVATE
  src
)

file(GLOB_RECURSE sources LIST_DIRECTORIES false "src/*.[hc]pp")

target_sources(${PROJECT_NAME}-example PRIVATE
  ${sources}
)
