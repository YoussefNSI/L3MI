#include "jeu.h"
#include <ostream>

std::string objet::to_string() const
{
    return "Pos(" + std::to_string(position().x()) + "," + std::to_string(position().y()) + ")"
            + "Tai(" + std::to_string(taille().w()) + "," + std::to_string(taille().h()) + ")";
}

bool objet::collision(const objet& o) const{
    return _p.x() < o._p.x() + o._t.w() && _p.x() + _t.w() > o._p.x() &&
               _p.y() < o._p.y() + o._t.h() && _p.y() + _t.h() > o._p.y();
}

std::ostream &operator<<(std::ostream &os, const objet &o){
    os << o.to_string();
    return os;
}

std::string objet_en_mouvement::to_string() const{
    return objet::to_string() + "Vit(" + std::to_string(vitesse().dx()) + "," + std::to_string(vitesse().dy());
}


std::string mur::to_string() const{
    return "Mur " + objet::to_string();
}

std::string bloc::to_string() const{
    TYPE t = type();
    if( t == TYPE::destructible)
        return "Bloc Destructible" + objet::to_string();
    return "Bloc Indestructible" + objet::to_string();
}

std::string balle::to_string() const{
    return "Balle " + objet_en_mouvement::to_string();
}

std::string raquette::to_string() const{
    return "Raquette " + objet_en_mouvement::to_string();
}


void raquette::modifier_direction(direction d){
    if(_d != d){
        switch(d){
        case direction::stop:
            modifier_v(vitesse_t(0,0));
            break;
        case direction::droite:
            modifier_v(vitesse_t(1,0));
            break;
        case direction::gauche:
            modifier_v(vitesse_t(-1,0));
            break;
        }
    }
}
