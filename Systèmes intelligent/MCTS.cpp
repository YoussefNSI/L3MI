#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>

class TreeNode
{
public:
    TreeNode *parent;
    std::vector<TreeNode *> children;
    int visit_count;
    double total_value;
    int move; // Le coup qui a mené à ce noeud (optionnel)

    TreeNode(TreeNode *p = nullptr, int m = -1)
        : parent(p), visit_count(0), total_value(0.0), move(m) {}

    ~TreeNode()
    {
        for (TreeNode *child : children)
            delete child;
    }

    bool is_fully_expanded() const
    {
        if (children.empty())
            return false;
        // À compléter: vérifier si tous les coups possibles sont générés
        return true;
    }

    TreeNode *get_child_with_max_score()
    {
        TreeNode *best = nullptr;
        double max_score = -std::numeric_limits<double>::max();

        for (TreeNode *child : children)
        {
            double score = child->total_value / (child->visit_count + 1e-5);
            if (score > max_score)
            {
                max_score = score;
                best = child;
            }
        }
        return best;
    }
};

class MCTS
{
private:
    TreeNode *root;
    const double exploration_weight = 1.4142; // sqrt(2)

    TreeNode *select(TreeNode *node)
    {
        while (!is_terminal(node))
        {
            if (node->is_fully_expanded())
            {
                node = best_ucb_child(node);
            }
            else
            {
                return expand(node);
            }
        }
        return node;
    }

    TreeNode *expand(TreeNode *node)
    {
        // Génère les coups possibles (à adapter)
        std::vector<int> possible_moves = get_possible_moves(node);

        for (int move : possible_moves)
        {
            TreeNode *child = new TreeNode(node, move);
            node->children.push_back(child);
        }
        return node->children.front(); // À adapter selon la stratégie
    }

    double simulate(TreeNode *node)
    {
        // Simulation aléatoire (à remplacer par la logique du jeu)
        return rand() % 100; // Valeur de simulation factice
    }

    void backpropagate(TreeNode *node, double result)
    {
        while (node != nullptr)
        {
            node->visit_count++;
            node->total_value += result;
            node = node->parent;
        }
    }

    TreeNode *best_ucb_child(TreeNode *node)
    {
        TreeNode *best = nullptr;
        double best_score = -std::numeric_limits<double>::max();

        for (TreeNode *child : node->children)
        {
            double exploitation = child->total_value / (child->visit_count + 1e-5);
            double exploration = exploration_weight *
                                 sqrt(log(node->visit_count + 1) / (child->visit_count + 1e-5));
            double score = exploitation + exploration;

            if (score > best_score)
            {
                best_score = score;
                best = child;
            }
        }
        return best;
    }

public:
    MCTS() : root(new TreeNode()) { srand(time(0)); }

    ~MCTS()
    {
        delete root;
    }

    void run_iteration()
    {
        TreeNode *leaf = select(root);
        double simulation_result = simulate(leaf);
        backpropagate(leaf, simulation_result);
    }

    // À implémenter selon les règles du jeu
    bool is_terminal(TreeNode *node)
    {
        // Vérifie si l'état est terminal
        return false;
    }

    std::vector<int> get_possible_moves(TreeNode *node)
    {
        // Génère les coups possibles depuis cet état
        return {1, 2, 3}; // Exemple
    }

    void print_stats() const
    {
        std::cout << "Root - Visits: " << root->visit_count
                  << " Value: " << root->total_value << "\n";
        for (TreeNode *child : root->children)
        {
            std::cout << "  Move " << child->move
                      << " - Visits: " << child->visit_count
                      << " Value: " << child->total_value << "\n";
        }
    }
};

int main()
{
    MCTS mcts;
    for (int i = 0; i < 1000; ++i)
        mcts.run_iteration();
    mcts.print_stats();
    return 0;
}