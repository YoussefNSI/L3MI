#ifndef TAILLEPOSITION_HH
#define TAILLEPOSITION_HH

#endif // TAILLEPOSITION_HH

#pragma once

using coordonnee = unsigned int;

class position {
public:
    position(coordonnee x, coordonnee y);
    position(position const& p)=default;
    coordonnee x() const;
    coordonnee y() const;
    void setx(coordonnee x);
    void sety(coordonnee y);
    bool comparaison(position const& p) const;
private:
    coordonnee x_;
    coordonnee y_;
};

class taille {
public:
    taille(coordonnee largeur, coordonnee hauteur);
    taille(taille const& p)=default;
    coordonnee largeur() const;
    coordonnee hauteur() const;
    void setlargeur(coordonnee largeur);
    void sethauteur(coordonnee hauteur);
    bool comparaison(taille const& p) const;
private:
    coordonnee largeur_;
    coordonnee hauteur_;
};
