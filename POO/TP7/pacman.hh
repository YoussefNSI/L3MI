#ifndef PACMAN_HH
#define PACMAN_HH

#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <memory> // Ajout de l'inclusion pour les smart pointers

#endif // PACMAN_HH

enum class direction
{
    stop,
    droite,
    gauche,
    haut,
    bas
};

class position
{
public:
    position(unsigned int x, unsigned int y);
    unsigned int x() const;
    unsigned int y() const;
    std::string to_string() const;
    bool operator==(const position &p) const;
    bool operator!=(const position &p) const;
    friend std::ostream &operator<<(std::ostream &os, const position &p);

private:
    unsigned int _x;
    unsigned int _y;
};

class taille
{
public:
    taille(unsigned int largeur, unsigned int hauteur);
    unsigned int w() const;
    unsigned int h() const;
    std::string to_string() const;
    friend std::ostream &operator<<(std::ostream &os, const taille &t);

private:
    unsigned int _largeur;
    unsigned int _hauteur;
};

class element
{
public:
    element(position pos, taille t);
    virtual ~element();
    position pos() const;
    taille tai() const;
    void setpos(position p);
    virtual char typeobjet() const;
    friend std::ostream &operator<<(std::ostream &os, const element &e);
    bool contient(const element &e) const;
    bool intersection(const element &e) const;

private:
    position _pos;
    taille _t;
    char _typeobjet;
};

class pacman : public element
{
public:
    pacman(position pos, direction d);
    direction deplacement() const;
    void setdir(direction d);
    char typeobjet() const override;
    bool invincible() const;
    void decrementerinvincible();
    void devenirinvincible();

private:
    direction _d;
    int _invicibilite;
};

class fantome : public element
{
public:
    fantome(position pos, direction d);
    direction deplacement() const;
    void setdir(direction d);
    char typeobjet() const override;

private:
    direction _d;
};

class mur : public element
{
public:
    mur(position pos, taille t);
    char typeobjet() const override;
};

class pacgommes : public element
{
public:
    pacgommes(position pos);
    char typeobjet() const override;
};

class exceptionjeu : public std::exception
{
public:
    exceptionjeu(std::string message);
    const char *what() const noexcept override;

private:
    std::string _message;
};

enum class etat
{
    encours,
    defaite,
    victoire
};

class jeu
{
public:
    jeu(std::vector<std::shared_ptr<element>> elements);
    jeu(const jeu &jeu);
    jeu &operator=(const jeu &jeu);
    std::ostream &afficher(std::ostream &os) const;
    void ajouter(std::shared_ptr<element> e);
    void ajouterfantome(int e);
    std::shared_ptr<pacman> accespacman();
    void directionjoueur(direction d);
    void changerdirectionfantomes();
    void tourdejeu();

private:
    std::vector<std::shared_ptr<element>> _elements;
    etat _etat;
    std::shared_ptr<pacman> _pacman;

    void appliquerdeplacementcollisionmur();
    void appliquerdeplacementcontact();
    void appliquerdeplacementmanger();
};