cmake_minimum_required(VERSION 3.18.0)
project(snake LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	add_compile_options(-Wall -Wpedantic)
endif()

find_package(SFML COMPONENTS graphics window system)
add_executable(snake snake.cc main.cc)
target_link_libraries(snake PRIVATE sfml-graphics sfml-window sfml-system)
target_compile_features(snake PRIVATE cxx_std_17)
