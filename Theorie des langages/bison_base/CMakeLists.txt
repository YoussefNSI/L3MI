project(bison_base)
cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS off)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wpedantic)
endif()

find_package(BISON)
find_package(FLEX)

BISON_TARGET(parser parser.yy ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
FLEX_TARGET(scanner scanner.ll  ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp)
ADD_FLEX_BISON_DEPENDENCY(scanner parser)


include_directories(. ${CMAKE_CURRENT_BINARY_DIR})


add_executable(${PROJECT_NAME}
    main.cc
    driver.hh
    driver.cc
    parser.yy
    scanner.hh
    scanner.ll
    ${BISON_parser_OUTPUTS}
    ${FLEX_scanner_OUTPUTS}
)
