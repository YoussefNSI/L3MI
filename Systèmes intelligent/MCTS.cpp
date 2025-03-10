#include "MCTS.h"

// Noeud class implementations
Noeud::Noeud(int state, std::shared_ptr<Noeud> p, int coup)
    : state_val(state), parent(p), coup_from_parent(coup), 
      visite_count(0), total_reward(0.0) {}

bool Noeud::est_terminal(const Jeu& game) const {
    return game.terminal();
}

bool Noeud::est_completement_developpe(const Jeu& game) {
    return coups_possibles.size() >= game.nb_coups();
}

double Noeud::calcul_ucb(double exploration_param) const {
    if (visite_count == 0) return std::numeric_limits<double>::max();
    return (total_reward / visite_count) + 
           exploration_param * sqrt(log(parent.lock()->visite_count) / visite_count);
}

// MCTS class implementations
MCTS::MCTS() : racine(std::make_shared<Noeud>(0)) {}

void MCTS::reconstruire_etat(Jeu& game, int state_val) {
    game.reset();
    std::vector<int> coups;
    while (state_val != 0) {
        int coup = state_val % 10;
        coups.insert(coups.begin(), coup);
        state_val = (state_val - coup) / 10;
    }
    for (int c : coups) game.joue(c);
}

std::shared_ptr<Noeud> MCTS::selection(std::shared_ptr<Noeud> noeud) {
    while (!noeud->est_terminal(jeu_courant)) {
        if (!noeud->est_completement_developpe(jeu_courant)) {
            return expansion(noeud);
        } else {
            noeud = meilleur_enfant_ucb(noeud);
        }
    }
    return noeud;
}

std::shared_ptr<Noeud> MCTS::expansion(std::shared_ptr<Noeud> noeud) {
    Jeu game_copy;
    reconstruire_etat(game_copy, noeud->state_val);
    
    // Générer les coups possibles
    int nb = game_copy.nb_coups();
    for (int coup = 1; coup <= nb; ++coup) {
        if (game_copy.coup_licite(coup)) {
            noeud->coups_possibles.push_back(coup);
        }
    }
    
    // Créer un nouvel enfant
    int nouveau_state = noeud->state_val * 10 + noeud->coups_possibles[0];
    auto enfant = std::make_shared<Noeud>(nouveau_state, noeud, noeud->coups_possibles[0]);
    noeud->enfants.push_back(enfant);
    return enfant;
}

double MCTS::simulation(std::shared_ptr<Noeud> noeud) {
    Jeu simulation_game;
    reconstruire_etat(simulation_game, noeud->state_val);
    
    while (!simulation_game.terminal()) {
        int nb = simulation_game.nb_coups();
        if (nb <= 0) break;
        int coup = simulation_game.random(nb);
        simulation_game.joue(coup);
    }
    
    if (simulation_game.victoire()) return 1.0;
    if (simulation_game.pat()) return 0.5;
    return 0.0;
}

void MCTS::retropropagation(std::shared_ptr<Noeud> noeud, double resultat) {
    while (noeud) {
        noeud->visite_count++;
        noeud->total_reward += resultat;
        noeud = noeud->parent.lock();
    }
}

std::shared_ptr<Noeud> MCTS::meilleur_enfant_ucb(std::shared_ptr<Noeud> parent) {
    std::shared_ptr<Noeud> meilleur;
    double max_score = -std::numeric_limits<double>::max();
    
    for (auto& enfant : parent->enfants) {
        double score = enfant->calcul_ucb(exploration_param);
        if (score > max_score) {
            max_score = score;
            meilleur = enfant;
        }
    }
    return meilleur;
}

void MCTS::effectuer_recherche(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        auto noeud = selection(racine);
        double resultat = simulation(noeud);
        retropropagation(noeud, resultat);
    }
}

int MCTS::meilleur_coup() {
    int meilleur = -1;
    int max_visites = -1;
    
    for (auto& enfant : racine->enfants) {
        if (enfant->visite_count > max_visites) {
            max_visites = enfant->visite_count;
            meilleur = enfant->coup_from_parent;
        }
    }
    return meilleur;
}
