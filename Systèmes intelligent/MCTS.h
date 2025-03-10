#pragma once

#include "jeu.h"
#include <memory>
#include <vector>
#include <cmath>
#include <ctime>

class Noeud {
public:
    int state_val;
    int visite_count;
    double total_reward;
    std::vector<std::shared_ptr<Noeud>> enfants;
    std::vector<int> coups_possibles;
    std::weak_ptr<Noeud> parent;
    int coup_from_parent;

    Noeud(int state = 0, std::shared_ptr<Noeud> p = nullptr, int coup = -1);

    bool est_terminal(const Jeu& game) const;
    bool est_completement_developpe(const Jeu& game);
    double calcul_ucb(double exploration_param) const;
};

class MCTS {
private:
    std::shared_ptr<Noeud> racine;
    const double exploration_param = 1.4142;
    Jeu jeu_courant;

    void reconstruire_etat(Jeu& game, int state_val);
    std::shared_ptr<Noeud> selection(std::shared_ptr<Noeud> noeud);
    std::shared_ptr<Noeud> expansion(std::shared_ptr<Noeud> noeud);
    double simulation(std::shared_ptr<Noeud> noeud);
    void retropropagation(std::shared_ptr<Noeud> noeud, double resultat);
    std::shared_ptr<Noeud> meilleur_enfant_ucb(std::shared_ptr<Noeud> parent);

public:
    MCTS();
    void effectuer_recherche(int iterations);
    int meilleur_coup();
};
