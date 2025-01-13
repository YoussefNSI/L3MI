#ifndef PACMAN_HH
#define PACMAN_HH

#include <string>
#include <iostream>
#include <vector>

#endif // PACMAN_HH
#include <stdexcept>

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
    position(unsigned int x, unsigned int y)
        : _x(x), _y(y) {}
    unsigned int x() const { return _x; }
    unsigned int y() const { return _y; }
    std::string to_string() const
    {
        return "(" + std::to_string(_x) + "," + std::to_string(_y) + ")";
    }
    bool operator==(const position &p) const
    {
        return _x == p._x && _y == p._y;
    }
    bool operator!=(const position &p) const
    {
        return !(*this == p);
    }
    friend std::ostream &operator<<(std::ostream &os, const position &p)
    {
        os << p.to_string();
        return os;
    }

private:
    unsigned int _x;
    unsigned int _y;
};

class taille
{
public:
    taille(unsigned int largeur, unsigned int hauteur)
        : _largeur(largeur), _hauteur(hauteur) {}
    unsigned int w() const { return _largeur; }
    unsigned int h() const { return _hauteur; }
    std::string to_string() const
    {
        return "(" + std::to_string(_largeur) + "," + std::to_string(_hauteur) + ")";
    }
    friend std::ostream &operator<<(std::ostream &os, const taille &t)
    {
        os << t.to_string();
        return os;
    }

private:
    unsigned int _largeur;
    unsigned int _hauteur;
};

class element
{
public:
    element(position pos, taille t)
        : _pos(pos), _t(t) {}
    position pos() const { return _pos; }
    taille tai() const { return _t; }
    void setpos(position p)
    {
        _pos = p;
    }
    virtual char typeobjet() const;
    friend std::ostream &operator<<(std::ostream &os, const element &e)
    {
        os << "Position : " << e._pos.to_string() << " Taille : " << e._t.to_string();
        return os;
    }
    bool contient(const element &e) const
    {
        return _pos.x() <= e._pos.x() && _pos.y() <= e._pos.y() &&
               (_pos.x() + _t.w()) >= (e._pos.x() + e._t.w()) &&
               (_pos.y() + _t.h()) >= (e._pos.y() + e._t.h());
    }
    bool intersection(const element &e) const
    {
        return _pos.x() < e._pos.x() + e._t.w() && _pos.x() + _t.w() > e._pos.x() &&
               _pos.y() < e._pos.y() + e._t.h() && _pos.y() + _t.h() > e._pos.y();
    }

private:
    position _pos;
    taille _t;
    char _typeobjet;
};

class pacman : public element
{
public:
    pacman(position pos, direction d)
        : element(pos, taille(13, 13)), _d(d), _invicibilite(0) {}
    direction deplacement() const { return _d; }
    void setdir(direction d) { _d = d; }
    char typeobjet() const override { return 'P'; }
    bool invincible() const { return _invicibilite > 0; }
    void decrementerinvincible()
    {
        if (_invicibilite > 0)
            _invicibilite--;
    }
    void devenirinvincible() { _invicibilite = 200; }

private:
    direction _d;
    int _invicibilite;
};

class fantome : public element
{
public:
    fantome(position pos, direction d)
        : element(pos, taille(20, 20)), _d(d) {}
    direction deplacement() const { return _d; }
    void setdir(direction d) { _d = d; }
    char typeobjet() const override { return 'F'; }

private:
    direction _d;
};

class mur : public element
{
public:
    mur(position pos, taille t)
        : element(pos, t)
    {
        if (t.w() < 10 || t.h() < 10)
        {
            throw std::invalid_argument("La largeur et la hauteur d'un mur doivent être toutes les deux supérieures ou égales à 10.");
        }
    }
    char typeobjet() const override { return 'M'; }
};

class exceptionjeu : public std::exception
{
public:
    exceptionjeu(std::string message) : _message(message) {}
    const char *what() const noexcept override { return _message.c_str(); }
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
    jeu(std::vector<element*> elements) : _elements(elements), _etat(etat::encours) {}
    jeu(const jeu &jeu) : _elements(jeu._elements), _etat(jeu._etat) {}
    jeu &operator=(const jeu &jeu)
    {
        _elements = jeu._elements;
        _etat = jeu._etat;
        return *this;
    }
    std::ostream& afficher(std::ostream& os) const{
        for (auto e : _elements)
        {
            os << *e << std::endl;
        }
        switch (_etat)
        {
            case etat::encours:
                os << "Partie en cours" << std::endl;
                break;
            case etat::defaite:
                os << "Partie perdue" << std::endl;
                break;
            case etat::victoire:
                os << "Partie gagnée" << std::endl;
                break;
        }
        return os;
    }
    void ajouter(element *e)
    {
        for (auto elem : _elements)
        {
            if (elem->intersection(*e))
            {
                throw exceptionjeu("L'élément à ajouter chevauche un autre élément.");
            }
        }
        _elements.push_back(e);
    }
private:
    std::vector<element *> _elements;
    etat _etat;
};


















