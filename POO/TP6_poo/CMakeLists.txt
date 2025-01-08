cmake_minimum_required(VERSION 3.5)

project(TP6_poo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS off)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wpedantic)
endif()

add_executable(TP6_poo main.cpp array.h array.cpp)

install(TARGETS TP6_poo
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
