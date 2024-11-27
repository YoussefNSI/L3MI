#pragma once
#include <vector>

using coordonnee = unsigned short;

enum class etatcellule {
	vivant,
	mort
};

enum class structure {
	oscillateurligne,
	floraison,
	planeur,
	oscillateurcroix
};

class grille {
public:
	grille(coordonnee l, coordonnee h);
	void vider();
	bool vivante(coordonnee x, coordonnee y) const;
	void generer(coordonnee x, coordonnee y);
	void tuer(coordonnee x, coordonnee y);
	void afficher() const;
	void ajouterstructure(structure s, coordonnee x, coordonnee y);
	unsigned int vivantes(coordonnee x, coordonnee y) const;
	void evolution(grille & result) const;
private:
	std::vector<etatcellule>::size_type indice(coordonnee x, coordonnee y) const;
	bool vivantesure(signed int x, signed int y) const;
private:
	coordonnee _largeur;
	coordonnee _hauteur;
	std::vector<etatcellule> _cellules;
};
