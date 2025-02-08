cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0148 OLD)

project(iANNS C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-w)

include_directories(${CMAKE_SOURCE_DIR}/anns/include)
include_directories(${CMAKE_SOURCE_DIR}/interval)

find_package(OpenMP REQUIRED) 

add_subdirectory(cplusplus)
add_subdirectory(experiment)