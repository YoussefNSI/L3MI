cmake_minimum_required(VERSION 3.5)

project(cc2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS off)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wpedantic)
endif()

add_executable(cc2 main.cpp taillepos.cc taillepos.hh jeu.cpp jeu.h)

install(TARGETS cc2
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
