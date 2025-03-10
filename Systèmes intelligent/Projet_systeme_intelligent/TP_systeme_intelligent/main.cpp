#include "jeu.h"
#include "MCTS.h"

#include <iostream>

int main() {

Jeu J;

std::shared_ptr<Noeud> n = std::make_shared<Noeud>(J,nullptr);


int score = n->roll_out(J);
std::cout << "score final : " << score << std::endl;

//std::cout << J.random(J.nb_coups());

//MCTS mcts;

// if (J.victoire() )
// {
    
// }
    


}