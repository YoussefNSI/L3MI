#include "grille.hh"
#include <iostream>

grille::grille(coordonnee l, coordonnee h)
	:_largeur(l), _hauteur(h), _cellules(l*h,etatcellule::mort) {
}

void grille::vider() {
	for (auto & c : _cellules)
		c = etatcellule::mort;
}

bool grille::vivante(coordonnee x, coordonnee y) const {
	return _cellules[indice(x,y)] == etatcellule::vivant;
}

void grille::generer(coordonnee x, coordonnee y) {
	_cellules[indice(x,y)] = etatcellule::vivant;
}

void grille::tuer(coordonnee x, coordonnee y) {
	_cellules[indice(x,y)] = etatcellule::mort;
}

void grille::afficher() const {
	for (coordonnee y=0; y<_hauteur; y++) {
		for (coordonnee x=0; x<_largeur; x++)
			std::cout << (vivante(x,y) ? "* " : "  ");
		std::cout << "\n";
	}
}

void grille::ajouterstructure(structure s, coordonnee x, coordonnee y) {
	switch (s) {
		case structure::oscillateurligne:
			generer(x,y); generer(x+1,y); generer(x+2,y);
			break;
		case structure::floraison:
			ajouterstructure(structure::oscillateurligne, x+1, y);
			ajouterstructure(structure::oscillateurligne, x+1, y+2);
			generer(x,y+1); generer(x+2,y+1); generer(x+4,y+1);
			break;
		case structure::planeur:
			ajouterstructure(structure::oscillateurligne, x, y);
			generer(x+2,y+1); generer(x+1,y+2);
			break;
		case structure::oscillateurcroix:
			ajouterstructure(structure::oscillateurligne, x, y+1);
			generer(x+1,y); generer(x+1,y+2);
			break;

	}
}

unsigned int grille::vivantes(coordonnee x, coordonnee y) const {
	unsigned int result(0);
	signed int sx(x), sy(y);
	for (signed int dx=-1; dx<=1; dx++)
		for (signed int dy=-1; dy<=1; dy++)
			if ((dx != 0) || (dy != 0))
				if (vivantesure(sx+dx,sy+dy))
					result++;
	return result;
}

void grille::evolution(grille & result) const {
	for (coordonnee y(0); y<_hauteur; ++y)
		for (coordonnee x(0); x<_largeur; ++x) {
			auto nbviv = vivantes(x,y);
			if (nbviv == 3)
				result.generer(x,y);
			else if ((nbviv <= 1) || (nbviv >= 4)) 
				result.tuer(x,y);
			else if (vivante(x,y))
				result.generer(x,y);
			else
				result.tuer(x,y);
		}
}

std::vector<etatcellule>::size_type grille::indice(coordonnee x, coordonnee y) const {
	return y * _largeur + x;
}

bool grille::vivantesure(signed int x, signed int y) const {
	return (x >= 0) && (y >= 0) && (x < _largeur) && (y < _hauteur) && (_cellules[indice(x,y)] == etatcellule::vivant);
}
