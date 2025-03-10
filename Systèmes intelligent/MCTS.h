// mcts.h
#pragma once
#include "jeu.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>

class TreeNode {
public:
    TreeNode* parent;
    std::vector<TreeNode*> children;
    int visit_count;
    double total_value;
    int move;       // Coup menant à ce nœud
    int state_val;  // Valeur de l'état du jeu
    std::vector<int> possible_moves;
    size_t next_move_idx;

    TreeNode(TreeNode* p = nullptr, int m = -1, int state = 0) 
        : parent(p), visit_count(0), total_value(0.0), move(m), 
          state_val(state), next_move_idx(0) {}

    ~TreeNode() {
        for(TreeNode* child : children) delete child;
    }

    bool is_fully_expanded() const {
        return next_move_idx >= possible_moves.size();
    }

    TreeNode* expand_next() {
        if (is_fully_expanded()) return nullptr;
        int move = possible_moves[next_move_idx++];
        int child_state = state_val * 10 + move;
        TreeNode* child = new TreeNode(this, move, child_state);
        children.push_back(child);
        return child;
    }
};

class MCTS {
private:
    TreeNode* root;
    const double exploration_weight = 1.4142; // sqrt(2)

    TreeNode* select(TreeNode* node) {
        while (true) {
            if (is_terminal(node)) return node;
            if (!node->is_fully_expanded()) return expand(node);
            node = best_ucb_child(node);
        }
    }

    TreeNode* expand(TreeNode* node) {
        if (node->possible_moves.empty()) {
            Jeu game;
            reconstruct_game_state(game, node->state_val);
            int nb = game.nb_coups();
            for (int i = 1; i <= nb; ++i) {
                if (game.coup_licite(i)) node->possible_moves.push_back(i);
            }
        }
        return node->expand_next();
    }

    double simulate(TreeNode* node) {
        Jeu game;
        reconstruct_game_state(game, node->state_val);
        while (!game.terminal()) {
            int nb = game.nb_coups();
            if (nb <= 0) break;
            int random_move = rand() % nb + 1;
            game.joue(random_move);
        }
        if (game.victoire()) return 1.0;
        else if (game.pat()) return 0.5;
        return 0.0;
    }

    void backpropagate(TreeNode* node, double result) {
        while (node != nullptr) {
            node->visit_count++;
            node->total_value += result;
            node = node->parent;
        }
    }

    TreeNode* best_ucb_child(TreeNode* node) {
        TreeNode* best = nullptr;
        double best_score = -std::numeric_limits<double>::max();
        for (TreeNode* child : node->children) {
            double exploitation = child->total_value / child->visit_count;
            double exploration = exploration_weight * 
                sqrt(log(node->visit_count) / child->visit_count);
            double score = exploitation + exploration;
            if (score > best_score) {
                best_score = score;
                best = child;
            }
        }
        return best;
    }

    void reconstruct_game_state(Jeu& game, int state_val) {
        game.reset();
        std::vector<int> moves;
        while (state_val != 0) {
            int move = state_val % 10;
            moves.insert(moves.begin(), move);
            state_val = (state_val - move) / 10;
        }
        for (int m : moves) game.joue(m);
    }

public:
    MCTS() : root(new TreeNode(nullptr, -1, RACINE)) { srand(time(0)); }
    ~MCTS() { delete root; }

    void run_iteration() {
        TreeNode* leaf = select(root);
        if (!is_terminal(leaf)) {
            double result = simulate(leaf);
            backpropagate(leaf, result);
        }
    }

    bool is_terminal(TreeNode* node) {
        Jeu game;
        reconstruct_game_state(game, node->state_val);
        return game.terminal();
    }

    int best_move() {
        int best_move = -1;
        int max_visits = -1;
        for (TreeNode* child : root->children) {
            if (child->visit_count > max_visits) {
                max_visits = child->visit_count;
                best_move = child->move;
            }
        }
        return best_move;
    }

    void print_stats() const {
        std::cout << "Root - Visits: " << root->visit_count 
                  << " Value: " << root->total_value << "\n";
        for (TreeNode* child : root->children) {
            std::cout << "Move " << child->move 
                      << " - Visits: " << child->visit_count
                      << " Value: " << child->total_value 
                      << " Avg: " << (child->total_value / child->visit_count) << "\n";
        }
    }
};