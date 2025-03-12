#include "MCTS.h"
#include <iostream>

int main() {
    MCTS mcts;
    mcts.effectuer_recherche(7500); // 1000 itérations

    int meilleur = mcts.meilleur_coup();
    std::cout << "Meilleur coup trouve : " << meilleur << std::endl;
    return 0;
}
