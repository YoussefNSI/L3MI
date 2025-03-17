#pragma once
#include "joueur.h"
#include "../MCTS.h"

class Joueur_MCTS : public Joueur
{
private:
    std::unique_ptr<MCTS> mcts_engine;
    int iterations;

public:
    Joueur_MCTS(std::string nom, bool joueur, int nb_iterations = 1000);

    void initialisation() override;
    void init_partie() override;
    char nom_abbrege() const override;
    void recherche_coup(Jeu jeu, int &coup) override;
};