cmake_minimum_required(VERSION 3.18.0)
project(tp8_1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS off)

find_package(Qt6Widgets REQUIRED)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-Wall -Wpedantic)
endif()

#add_executable(calculatrice calculatrice.cc calculatrice.hh)
add_executable(simon sequence.cc
    sequence.hh
    main.cc
    simon.hh
    simon.cc)
target_link_libraries(simon Qt6::Widgets)
