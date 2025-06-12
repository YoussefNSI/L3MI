#pragma once
#include "taillepos.hh"
#include <ostream>
#include <string>
#include <iostream>

enum class TYPE{
    destructible,
    indestructible
};



enum class direction
{
    stop,
    droite,
    gauche
};

class objet{
public:
    objet(const position_t& p, const taille_t& t) : _p(p), _t(t) {}
    position_t position() const { return _p; }
    taille_t taille() const { return _t; }
    void modifier_position(const position_t& p) { _p = p; }
    virtual bool detruit_quand_collision(const objet& o) const { return false; }
    virtual std::string to_string() const;
    friend std::ostream &operator<<(std::ostream &os, const objet &o);
    bool collision(const objet& o) const;
private:
    position_t _p;
    taille_t _t;
};

class objet_en_mouvement : public objet{
public:
    objet_en_mouvement(const position_t& p, const taille_t& t, const vitesse_t& v) : objet(p,t), _v(v) {}
    objet_en_mouvement(const objet_en_mouvement&)=default;
    ~objet_en_mouvement() = default;
    vitesse_t vitesse() const{ return _v; }
    void modifier_v(const vitesse_t& v) { _v = v; }
    bool detruit_quand_collision(const objet& o) const override { return true; }
    std::string to_string() const override;
    position_t calculer_deplacement(const vitesse_t& v) const { return (position() + v);}
    void appliquer_deplacement_simple() { modifier_position(calculer_deplacement(_v)); }



private:
    vitesse_t _v;
};

class mur : public objet{
public:
    mur(const position_t& p, const taille_t& t) : objet(p, t) {}
    std::string to_string() const override;
};

class bloc : public objet{
public:
    bloc(const position_t& p, TYPE t) : objet(p, taille_t(20,10)), _type(t) {}
    TYPE type() const{ return _type; }
    std::string to_string() const override;
private:
    TYPE _type;
};

class balle : public objet_en_mouvement{
public:
    balle(const position_t& p, const vitesse_t& v) : objet_en_mouvement(p, taille_t(5,5), v) {}
    std::string to_string() const override;

private:

};

class raquette : public objet_en_mouvement{
public:
    raquette(const position_t& p, const taille_t& t, int largeur)
        : objet_en_mouvement(p, t, vitesse_t(-1,0)), _d(direction::stop), _h(5), _l(largeur) {}
    int h() const {return _h;}
    int l() const {return _l;}
    void modifier_direction(direction d);
    std::string to_string() const override;

private:
    direction _d;
    int _h;
    int _l;

};






