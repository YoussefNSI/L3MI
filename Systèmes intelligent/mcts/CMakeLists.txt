cmake_minimum_required(VERSION 3.16)

project(mcts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(MCTS main.cc jeu.cpp jeu.h MCTS.cpp MCTS.h)



include(GNUInstallDirs)
install(TARGETS MCTS
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
