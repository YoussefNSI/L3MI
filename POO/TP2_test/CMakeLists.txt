cmake_minimum_required(VERSION 3.16)

project(TP2_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS off)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wpedantic)
endif()


add_executable(TP2_test main.cc)
add_executable(sequence sequence.hh sequence.cc)
add_executable(sequence2 sequence_2.hh sequence_2.cc)
add_executable(tailleposition tailleposition.hh tailleposition.cc)

include(GNUInstallDirs)
install(TARGETS TP2_test
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
