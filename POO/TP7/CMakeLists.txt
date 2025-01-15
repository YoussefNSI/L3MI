cmake_minimum_required(VERSION 3.18.0)

project(TP7 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS off)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wpedantic)
endif()

find_package(SFML COMPONENTS graphics window system REQUIRED)
target_link_libraries(TP7 PRIVATE sfml-graphics sfml-window sfml-system)
target_compile_features(TP7 PRIVATE cxx_std_17)

add_executable(TP7 main.cc ensemble.hh ensemble.cc)
add_executable(TP7_2 main2.cc pacman.hh pacman.cc)

include(GNUInstallDirs)
install(TARGETS TP7 TP7_2
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)


