cmake_minimum_required(VERSION 3.16)

project(TP2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(TP2 main.cc
    tailleposition.hh
    tailleposition.cc
    sequence.hh
    sequence.cc
)
add_executable(test_vector main_vect.cc)
add_executable(grille main_grille.cc grille.cc grille.hh)

include(GNUInstallDirs)
install(TARGETS TP2
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
