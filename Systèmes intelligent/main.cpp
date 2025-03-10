#include "MCTS.h"
#include "jeu.h"
#include <iostream>



int main() {
    MCTS mcts;
    for (int i = 0; i < 1000; ++i) mcts.run_iteration();
    std::cout << "Best move: " << mcts.best_move() << std::endl;
    return 0;
}