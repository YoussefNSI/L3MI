#include "tailleposition.hh"

position::position(coordonnee x, coordonnee y)
	:_x(x), _y(y) {
}

// Il est inutile de définir ce constructeur par recopie car le code que nous
// écrivons ici n'est rien d'autre que le comportement du constructeur par
// recopie implicite.
/*
position::position(position const & p)
	:_x(p._x), _y(p._y) {
}
*/

coordonnee position::x() const {
	return _x;
}

coordonnee position::y() const {
	return _y;
}

void position::setx(coordonnee x) {
	_x = x;
}

void position::sety(coordonnee y) {
	_y = y;
}

bool position::comparaison(position const & p) const {
	return (_x == p._x) && (_y == p._y);
}

taille::taille(coordonnee largeur, coordonnee hauteur)
	:_largeur(largeur), _hauteur(hauteur) {
}

coordonnee taille::largeur() const {
	return _largeur;
}

coordonnee taille::hauteur() const {
	return _hauteur;
}

void taille::setlargeur(coordonnee largeur) {
	_largeur = largeur;
}

void taille::sethauteur(coordonnee hauteur) {
	_hauteur = hauteur;
}

bool taille::comparaison(taille const & p) const {
	return (_largeur == p._largeur) && (_hauteur == p._hauteur);
}
