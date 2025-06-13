#include "billet.h"

double Billet::prix_total() const {
    double mult = (_type == Billet::Type::AllerRetour) ? 2.0 : 1.0;
    return _prixBase * mult * _nbVoyageurs;
}

std::string Billet::tostring() const {
    return _depart + " - " + _arrivee + " - "
            + ((_type == Billet::Type::AllerRetour) ? "AR" : "AS") + " - "
           + std::to_string(_prixBase) + " - "
           + std::to_string(_nbVoyageurs) + " - "
           + std::to_string(prix_total());
}

int Billet::compteur = 0;
