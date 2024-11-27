#pragma once

using coordonnee = unsigned int;

class position {
public:
	position(coordonnee x, coordonnee y);
	// On peut déclarer explicitement un constructeur par recopie et donner
	// son code (cf. fichier .cc), le déclarer "=default" comme ici, ou
	// l'omettre (pour profiter du constructeur par recopie implicite).
	position(position const & p) =default;
	coordonnee x() const;
	coordonnee y() const;
	void setx(coordonnee x);
	void sety(coordonnee y);
	bool comparaison(position const & p) const;
private:
	coordonnee _x;
	coordonnee _y;
};

class taille {
public:
	taille(coordonnee largeur, coordonnee hauteur);
	taille(taille const & p) =default;
	coordonnee largeur() const;
	coordonnee hauteur() const;
	void setlargeur(coordonnee x);
	void sethauteur(coordonnee y);
	bool comparaison(taille const & p) const;
private:
	coordonnee _largeur;
	coordonnee _hauteur;
};

