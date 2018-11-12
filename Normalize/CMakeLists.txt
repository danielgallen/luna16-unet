cmake_minimum_required(VERSION 3.0)

project(Normalize)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(Normalize normalize.cxx)

target_link_libraries(Normalize ${ITK_LIBRARIES})
