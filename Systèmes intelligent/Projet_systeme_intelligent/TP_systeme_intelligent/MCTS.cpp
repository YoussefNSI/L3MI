#include "MCTS.h"


void Noeud::ajouterNoeud(std::shared_ptr<Noeud> n)
{
    
    noeuds_fils.push_back(n);
    if (n->parent) // important, la racine n'a pas de noeud parent (nullptr), tous les noeuds sauf la racine 
        n->parent = shared_from_this(); //  retourne un autre shared_ptr vers *this, sans augmenter le compteur
}



int Noeud::score(Jeu j)
{
    if ( j.victoire() ) return 1;
    else if ( j.pat() ) return 0;
    else return -1; // defaite
}

void Noeud::update(int score)
{
    ++compteur_scenario;
    gain_accumule += score;

    if (parent != nullptr)
    {
        parent->update(score);
    }

}




int Noeud::roll_out(Jeu j)
{
    int coup;
    while (!j.terminal()) // pas une feuille
    {
        coup =  j.random(j.nb_coups());
        std::cout << "coup joué : " << coup << std::endl;
        j.joue(coup);

        std::shared_ptr<Noeud> fils = std::make_shared<Noeud>(j,nullptr);
        ajouterNoeud(fils);

    }
    return score(j); 
}

