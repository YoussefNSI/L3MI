cmake_minimum_required(VERSION 3.16)

project(TP7 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS off)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wpedantic)
endif()

add_executable(TP7 main.cc ensemble.hh ensemble.cc)
add_executable(TP7_2 main2.cc pacman.hh pacman.cc)

find_package(SFML COMPONENTS graphics window system)
target_link_libraries(TP7_2 PRIVATE sfml-graphics sfml-window sfml-system)
target_compile_features(TP7_2 PRIVATE cxx_std_17)

include(GNUInstallDirs)
install(TARGETS TP7 TP7_2
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)


