#pragma once

#include "jeu.h"

#include <memory>
#include <vector>



// class MCTS
// {
//     public:
//         int descente();
//         int roll_out();
//         void update(std::vector<int> & chemin, int resultat);
    
//     private:
//         int gain_cumule;

// };

class Noeud : public std::enable_shared_from_this<Noeud>
{

    public:
        Noeud(Jeu j, std::shared_ptr<Noeud> p)
        : etat_de_la_partie(j),parent(p), compteur_scenario(0), gain_accumule(0) {};

        void fabriqueFils(Jeu j);

        void ajouterNoeud(std::shared_ptr<Noeud> n);

        int score(Jeu j);
        void update(int score);

        int roll_out(Jeu j);

        //     void afficherParent() {
        //     if (auto p = parent.lock()) {  // Convertit le weak_ptr en shared_ptr
        //         std::cout << "Parent trouvé" << std::endl;
        //     } else {
        //         std::cout << "Pas de parent (nullptr)" << std::endl;
        //     }
        // }

        // faire un if (jeu->terminal() ) score() ...


    private:
        Jeu etat_de_la_partie;
        std::shared_ptr<Noeud> parent;
        std::vector<std::shared_ptr<Noeud>> noeuds_fils;
        int compteur_scenario;
        int gain_accumule;





};
