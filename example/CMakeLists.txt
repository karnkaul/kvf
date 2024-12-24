include(FetchContent)

add_executable(${PROJECT_NAME}-example WIN32)

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

if(MSVC)
  target_link_options(${PROJECT_NAME}-example PRIVATE
    "/ENTRY:mainCRTStartup"
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL Windows AND CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  target_link_options(${PROJECT_NAME}-example PRIVATE
    -Xlinker /ENTRY:mainCRTStartup
  )
endif()
