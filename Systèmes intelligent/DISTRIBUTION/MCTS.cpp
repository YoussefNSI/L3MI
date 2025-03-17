#include "MCTS.h"
#include <limits>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stack>
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <regex>

Noeud::Noeud(int state, std::shared_ptr<Noeud> p, int coup)
    : state_val(state), visite_count(0), total_reward(0.0),
      parent(p), coup_from_parent(coup) {}

bool Noeud::est_terminal(const Jeu &game) const
{
    return game.terminal();
}

bool Noeud::est_completement_developpe(const Jeu &game)
{
    return coups_possibles.size() >= static_cast<std::vector<int>::size_type>(game.nb_coups());
}

double Noeud::calcul_ucb(double exploration_param) const
{
    if (visite_count == 0)
        return std::numeric_limits<double>::max();
    return (total_reward / visite_count) +
           exploration_param * sqrt(log(parent.lock()->visite_count) / visite_count);
}

MCTS::MCTS() : racine(std::make_shared<Noeud>(0)) {}

void MCTS::reconstruire_etat(Jeu &game, int state_val)
{
    game.reset();
    std::vector<int> coups;
    while (state_val != 0 && state_val != RACINE)
    {
        int coup = state_val % 10;
        coups.insert(coups.begin(), coup);
        state_val = (state_val - coup) / 10;
    }
    for (int c : coups)
        if (c > 0)
            game.joue(c);
}

std::shared_ptr<Noeud> MCTS::selection(std::shared_ptr<Noeud> noeud)
{
    while (noeud && !noeud->est_terminal(jeu_courant))
    {
        if (!noeud->est_completement_developpe(jeu_courant))
        {
            auto expanded = expansion(noeud);
            if (expanded)
                return expanded;
        }
        noeud = meilleur_enfant_ucb(noeud);
    }
    return noeud;
}

std::shared_ptr<Noeud> MCTS::expansion(std::shared_ptr<Noeud> noeud)
{
    Jeu game_copy;
    reconstruire_etat(game_copy, noeud->state_val);

    // Générer les coups possibles
    int nb = game_copy.nb_coups();
    if (nb <= 0)
        return nullptr;

    for (int coup = 1; coup <= nb; ++coup)
    {
        if (game_copy.coup_licite(coup))
        {
            if (std::find(noeud->coups_possibles.begin(),
                          noeud->coups_possibles.end(),
                          coup) == noeud->coups_possibles.end())
            {
                noeud->coups_possibles.push_back(coup);
            }
        }
    }

    // Créer un nouvel enfant
    if (!noeud->coups_possibles.empty())
    {
        int nouveau_state = noeud->state_val * 10 + noeud->coups_possibles[0];
        auto enfant = std::make_shared<Noeud>(nouveau_state, noeud, noeud->coups_possibles[0]);
        noeud->enfants.push_back(enfant);
        return enfant;
    }
    return nullptr;
}

double MCTS::simulation(std::shared_ptr<Noeud> noeud)
{
    Jeu simulation_game;
    if (noeud)
        reconstruire_etat(simulation_game, noeud->state_val);
    else
        return 0.0;

    while (!simulation_game.terminal())
    {
        int nb = simulation_game.nb_coups();
        if (nb <= 0)
            break;
        int coup = simulation_game.random(nb);
        simulation_game.joue(coup);
    }

    if (simulation_game.victoire())
        return 1.0;
    if (simulation_game.pat())
        return 0.5;
    return 0.0;
}

void MCTS::retropropagation(std::shared_ptr<Noeud> noeud, double resultat)
{
    while (noeud)
    {
        noeud->visite_count++;
        noeud->total_reward += resultat;
        noeud = noeud->parent.lock();
    }
}

std::shared_ptr<Noeud> MCTS::meilleur_enfant_ucb(std::shared_ptr<Noeud> parent)
{
    std::shared_ptr<Noeud> meilleur;
    double max_score = -std::numeric_limits<double>::max();

    for (auto &enfant : parent->enfants)
    {
        double score = enfant->calcul_ucb(exploration_param);
        if (score > max_score)
        {
            max_score = score;
            meilleur = enfant;
        }
    }
    return meilleur;
}

void MCTS::effectuer_recherche(int iterations)
{
    for (int i = 0; i < iterations; ++i)
    {
        auto noeud = selection(racine);
        double resultat = simulation(noeud);
        retropropagation(noeud, resultat);
    }
}

int MCTS::meilleur_coup()
{
    int meilleur = -1;
    int max_visites = -1;

    for (auto &enfant : racine->enfants)
    {
        if (enfant->visite_count > max_visites)
        {
            max_visites = enfant->visite_count;
            meilleur = enfant->coup_from_parent;
        }
    }
    return meilleur;
}


#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void MCTS::save_arbre(const std::string& filename) {
    json j;
    std::unordered_map<std::shared_ptr<Noeud>, int> node_ids;
    std::queue<std::shared_ptr<Noeud>> queue;
    int current_id = 0;

    // Sauvegarde de la racine
    queue.push(racine);
    node_ids[racine] = current_id++;

    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop();

        json node_json;
        node_json["state"] = node->state_val;
        node_json["visits"] = node->visite_count;
        node_json["reward"] = node->total_reward;
        node_json["coup"] = node->coup_from_parent;

        // Parent
        if (auto parent = node->parent.lock()) {
            node_json["parent"] = node_ids[parent];
        } else {
            node_json["parent"] = -1;
        }

        // Enfants
        std::vector<int> children;
        for (auto& child : node->enfants) {
            if (!node_ids.count(child)) {
                node_ids[child] = current_id++;
                queue.push(child);
            }
            children.push_back(node_ids[child]);
        }
        node_json["children"] = children;

        // Ajout au document
        j["nodes"][std::to_string(node_ids[node])] = node_json;
    }

    j["root_id"] = node_ids[racine];

    // Écriture fichier
    std::ofstream file(filename);
    file << j.dump(4);
    file.close();
}

void MCTS::load_arbre(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file");

    json j;
    file >> j;
    file.close();

    if (!j.contains("nodes") || !j.contains("root_id"))
        throw std::runtime_error("Format de fichier invalide");

    std::unordered_map<int, std::shared_ptr<Noeud>> node_map;
    const auto& nodes = j["nodes"];

    // Première passe : créer tous les noeuds
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        int id = std::stoi(it.key());
        const auto& node_json = it.value();

        auto node = std::make_shared<Noeud>(
            node_json["state"].get<int>(),
            nullptr,
            node_json["coup"].get<int>()
        );
        node->visite_count = node_json["visits"].get<int>();
        node->total_reward = node_json["reward"].get<double>();

        node_map[id] = node;
    }

    // Deuxième passe : lier les relations
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        int id = std::stoi(it.key());
        const auto& node_json = it.value();
        auto node = node_map[id];

        // Parent
        int parent_id = node_json["parent"].get<int>();
        if (parent_id != -1) {
            node->parent = node_map[parent_id];
        }

        // Enfants
        for (const auto& child_id : node_json["children"]) {
            node->enfants.push_back(node_map[child_id.get<int>()]);
        }
    }

    // Définir la racine
    int root_id = j["root_id"].get<int>();
    if (node_map.count(root_id)) {
        racine = node_map[root_id];
    } else {
        throw std::runtime_error("Noeud racine introuvable");
    }
}
