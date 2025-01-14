#include "pacman.hh"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cstdlib>


std::ostream& jeu::afficher(std::ostream &os) const{
        for (auto e : _elements)
        {
            os << *e << std::endl;
        }
        switch (_etat)
        {
        case etat::encours:
            os << "Partie en cours" << std::endl;
            break;
        case etat::defaite:
            os << "Partie perdue" << std::endl;
            break;
        case etat::victoire:
            os << "Partie gagnée" << std::endl;
            break;
        }
        return os;
    }

void jeu::ajouter(element *e)
{
    if (std::any_of(_elements.begin(), _elements.end(), [e](element *element)
                    { return element->intersection(*e); }))
    {
        throw exceptionjeu("L'élément à ajouter chevauche un autre élément.");
    }
    _elements.push_back(e);
}


