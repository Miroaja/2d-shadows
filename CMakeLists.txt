cmake_minimum_required(VERSION 4.0)
project(angular_mask_demo)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Werror")

include_directories(SYSTEM "glad/include")

add_executable(angular_mask main.cpp)

target_sources(
  angular_mask PRIVATE
  glad/src/gl.c
  main.cpp
)

add_library(glad STATIC
  glad/src/gl.c
)


find_package(OpenGL REQUIRED)

find_package(glfw3 REQUIRED)

target_link_libraries(angular_mask
    OpenGL::GL
    glfw
)
