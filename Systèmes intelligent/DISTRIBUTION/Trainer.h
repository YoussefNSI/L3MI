#pragma once
#include "MCTS.h"
#include "jeu.h"

class MCTSTrainer {
private:
    int num_games;
    int simulations_per_move;

public:
    MCTSTrainer(int games = 1000, int sims = 100)
        : num_games(games), simulations_per_move(sims) {}

    MCTS mcts;
    void train() {
        mcts.enable_entrainement();
        
        for(int i = 0; i < num_games; ++i) {
            Jeu game;
            self_play(game);
            
            if(i % 100 == 0) {
                std::cout << "Game " << i << " completed" << std::endl;
                //mcts.save_arbre("checkpoint_" + std::to_string(i) + ".tree");
            }
        }
    }

    void self_play(Jeu& game) {
        while(!game.terminal()) {
            mcts.reset(game.get_etat());
            mcts.effectuer_recherche(simulations_per_move);
            
            int best_move = mcts.meilleur_coup();
            game.joue(best_move);
            
            // Mise à jour progressive avec résultat final
            update_tree_with_final_result(game);
        }
    }

    void update_tree_with_final_result(Jeu& final_state) {
        double result = final_state.victoire() ? 1.0 : 0.0;
        auto node = mcts.get_root();
        mcts.retropropagation(node, result);
    }
};