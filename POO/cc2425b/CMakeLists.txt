cmake_minimum_required(VERSION 3.18.0)
project(breakout LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	add_compile_options(-Wall -Wpedantic)
endif()

find_package(SFML COMPONENTS graphics window system)
add_executable(breakout taillepos.cc breakout.cc main.cc breakout.hh taillepos.hh)
target_link_libraries(breakout PRIVATE sfml-graphics sfml-window sfml-system)
target_compile_features(breakout PRIVATE cxx_std_17)
