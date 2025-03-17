double MCTS::simulation(std::shared_ptr<Noeud> noeud) {
    Jeu simulation_game;
    if (noeud) reconstruire_etat(simulation_game, noeud->state_val);

    while (!simulation_game.terminal()) {
        // Recherche de coups gagnants/urgents
        int best_coup = -1;
        for (int coup = 1; coup <= simulation_game.nb_coups(); coup++) {
            if (!simulation_game.coup_licite(coup)) continue;
            
            Jeu copie = simulation_game;
            copie.joue(coup);
            if (copie.victoire()) {
                best_coup = coup;
                break;
            }
        }

        if (best_coup != -1) {
            simulation_game.joue(best_coup);
            return 1.0; // Victoire immédiate
        }

        // Sinon, coup aléatoire
        int nb = simulation_game.nb_coups();
        if (nb <= 0) break;
        simulation_game.joue(simulation_game.random(nb));
    }
    return simulation_game.victoire() ? 1.0 : 0.0;
}

std::shared_ptr<Noeud> MCTS::expansion(std::shared_ptr<Noeud> noeud) {
    Jeu game_copy;
    reconstruire_etat(game_copy, noeud->state_val);
    
    std::vector<int> legal_moves;
    for (int coup = 1; coup <= game_copy.nb_coups(); ++coup) {
        if (game_copy.coup_licite(coup)) {
            legal_moves.push_back(coup);
        }
    }

    // Ajouter tous les nouveaux coups possibles
    for (int coup : legal_moves) {
        if (std::find(noeud->coups_possibles.begin(), 
                      noeud->coups_possibles.end(), coup) == noeud->coups_possibles.end()) {
            noeud->coups_possibles.push_back(coup);
            int nouveau_state = noeud->state_val * 10 + coup;
            auto enfant = std::make_shared<Noeud>(nouveau_state, noeud, coup);
            noeud->enfants.push_back(enfant);
        }
    }

    return noeud->enfants.empty() ? nullptr : noeud->enfants.front();
}

class MCTS {
private:
    double get_exploration_param() const {
        return 2.5 * exp(-racine->visite_count / 1000.0);
    }
    // Utiliser cette méthode dans calcul_ucb
};

double Noeud::calcul_ucb(double exploration_param) const {
    if (visite_count == 0) return std::numeric_limits<double>::max();
    double parent_visits = parent.lock()->visite_count;
    return (total_reward / visite_count) + 
           exploration_param * sqrt(log(parent_visits + 1) / (visite_count + 1e-5);
}


class MCTS {
private:
    std::unordered_map<int, std::shared_ptr<Noeud>> node_pool;
    
    std::shared_ptr<Noeud> create_node(int state, std::shared_ptr<Noeud> parent, int coup) {
        if (node_pool.count(state)) return node_pool[state];
        auto node = std::make_shared<Noeud>(state, parent, coup);
        node_pool[state] = node;
        return node;
    }
};

class MCTSTrainer {
private:
    std::vector<Experience> replay_buffer;

    void self_play(Jeu& game) {
        std::vector<std::shared_ptr<Noeud>> history;
        
        while(!game.terminal()) {
            mcts.reset(game.get_etat());
            mcts.effectuer_recherche(simulations_per_move);
            history.push_back(mcts.get_root());
            
            int best_move = mcts.meilleur_coup();
            game.joue(best_move);
        }
        
        // Rétropropagation avec discount factor
        double result = game.victoire() ? 1.0 : -1.0;
        for (auto it = history.rbegin(); it != history.rend(); ++it) {
            mcts.retropropagation(*it, result);
            result *= 0.9; // Discount factor
        }
    }
};
