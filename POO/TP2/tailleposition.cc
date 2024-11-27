#include "tailleposition.hh"

position::position(coordonnee x, coordonnee y) :
    x_(x), y_(y) {
}

coordonnee position::x() const {
    return x_;
}

coordonnee position::y() const {
    return y_;
}

void position::setx(coordonnee x) {
    x_ = x;
}

void position::sety(coordonnee y) {
    y_ = y;
}

bool position::comparaison(position const& p) const {
    return x_ == p.x() && y_ == p.y();
}

taille::taille(coordonnee largeur, coordonnee hauteur) :
    largeur_(largeur), hauteur_(hauteur) {
}

coordonnee taille::largeur() const {
    return largeur_;
}

coordonnee taille::hauteur() const {
    return hauteur_;
}

void taille::setlargeur(coordonnee largeur) {
    largeur_ = largeur;
}

void taille::sethauteur(coordonnee hauteur) {
    hauteur_ = hauteur;
}

bool taille::comparaison(taille const& p) const {
    return largeur_ == p.largeur() && hauteur_ == p.hauteur();
}
