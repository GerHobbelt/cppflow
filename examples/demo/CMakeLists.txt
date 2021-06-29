cmake_minimum_required(VERSION 3.10)
project(example)

set(CMAKE_CXX_STANDARD 17)


link_directories(D:\\myself\\code\\cpp\\cppflow2\\lib)
link_directories(/mnt/lib)


add_executable(example main.cpp )

add_library(captcha   SHARED main_lib.cpp)
target_include_directories(example PRIVATE ../../include)
target_include_directories(captcha PRIVATE ../../include ./)
if(WIN32)
    message("WIN32")
    target_link_libraries (example tensorflow  )
    target_link_libraries (captcha tensorflow  )
else()
    message("linux")
    target_link_libraries (example tensorflow tensorflow_framework pthread)
    target_link_libraries (captcha tensorflow  tensorflow_framework )
endif()

