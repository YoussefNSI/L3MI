project(flex_base)
cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS off)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wpedantic)
endif()

find_package(FLEX)

FLEX_TARGET(scanner scanner.ll  ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp)

include_directories(. ${CMAKE_CURRENT_BINARY_DIR})

add_executable(${PROJECT_NAME}
    scanner.ll
    ${FLEX_scanner_OUTPUTS}
)
