cmake_minimum_required(VERSION 3.10)
project(example)


set(CMAKE_CXX_STANDARD 17)

add_executable(example main.cpp odr.cpp)
target_include_directories(example PRIVATE ../../include)
target_link_libraries(example tensorflow)
