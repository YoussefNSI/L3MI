#ifndef PACMAN_HH
#define PACMAN_HH

#include <string>

#endif // PACMAN_HH

enum class direction {
    stop,
    droite,
    gauche,
    haut,
    bas
};

class position {
public:
    position(unsigned int x, unsigned int y)
        : _x(x), _y(y) {}
    unsigned int x() const { return _x; }
    unsigned int y() const { return _y; }
    std::string to_string() const {
        return "(" + std::to_string(_x) + "," + std::to_string(_y) + ")";
    }
    bool operator==(const position& p) const {
        return _x == p._x && _y == p._y;
    }
    bool operator!=(const position& p) const {
        return !(*this == p);
    }
    void flux(std::ostream& os) const {
        os << to_string();
    }
private:
    unsigned int _x;
    unsigned int _y;
};

class taille{
    taille(unsigned int largeur, unsigned int hauteur)
        : _largeur(largeur), _hauteur(hauteur) {}
    unsigned int w() const { return _largeur; }
    unsigned int h() const { return _hauteur; }
    std::string to_string() const {
        return "(" + std::to_string(_largeur) + "," + std::to_string(_hauteur) + ")";
    }
    void flux(std::ostream& os) const {
        os << to_string();
    }
private:
    unsigned int _largeur;
    unsigned int _hauteur;
};

class element{
public:
    element(position pos, taille t)
        : _pos(pos), _t(t) {}
    position pos() const { return _pos; }
    taille t() const { return _t; }
    void setpos(position p){
        _pos = p;
    }
    virtual char typeobjet() const { return _typeobjet; }
private:
    position _pos;
    taille _t;
    char _typeobjet;
};

class pacman : public element {
public:
    pacman(position pos, direction d)
        : element(pos, taille(13, 13)) {}
};











