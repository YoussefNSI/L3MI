#ifndef ELEMENT_HH
#define ELEMENT_HH

#include "tailleposition.hh"
#include <string>
#endif // ELEMENT_HH

class element:
    public taille, public position {
    element(coordonnee x, coordonnee y, coordonnee largeur, coordonnee hauteur);
    element(element const & e) =default;
public:
    void setposition(coordonnee x, coordonnee y);
    void settaille(coordonnee largeur, coordonnee hauteur);
    taille const & gettaille() const;
    position const & getposition() const;
    std::string tostring() const;
    bool contientposition(position const & p) const;
private:
    taille _taille;
    position _position;
};


