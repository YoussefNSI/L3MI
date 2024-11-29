#ifndef SNAKE_HH
#define SNAKE_HH

#include <list>
#include <string>
#include <vector>
#endif // SNAKE_HH


using coord = unsigned int;

class position
{
    public:
        position(coord x, coord y);
        position(position const & p) =default;
        coord x() const;
        coord y() const;
        void setx(coord x);
        void sety(coord y);
        std::string tostring() const;
        bool egal(position const & p) const;
        bool adjacent(position const & p) const;
    private:
        coord _x;
        coord _y;
};

enum class typecase{
    serpent1,
    serpent2,
    bonus,
    obstacle,
    vide
};

enum class direction{
    haut,
    bas,
    gauche,
    droite
};

class affichage{
public:
    affichage(coord largeur, coord hauteur)
        : _cases(largeur * hauteur, typecase::vide)
        , _largeur(largeur)
        , _hauteur(hauteur) {}
    void vider();
    typecase acces(position const &p) const { return _cases.at(p.x() + p.y() * _largeur); };
    void fixer(position const &p, typecase t) { _cases.at(p.x() + p.y() * _largeur) = t; };
    void afficher(std::ostream &os) const;
private:
    std::vector<typecase> _cases;
    coord _largeur;
    coord _hauteur;
};


class element{
public:
    element(position const & p);
    virtual ~element() =default;
    position const & pos() const;
    virtual typecase type() const =0;

protected:
    void modifier_position(position const & p);
private:
    position _pos;
};

class enmouvement : public element{
public:
    enmouvement(position const& p, direction d);
    direction dir() const { return _dir; };
    virtual void appliquer_deplacement();

protected:
    void modifier_direction(direction d) { _dir = d; };
private:
    direction _dir;
};


class bonus : public enmouvement{
public:
    bonus(position const&p, direction dir, unsigned int points);
    typecase type() const override { return typecase::bonus; };
    unsigned int points() const { return _points; };
private:
    unsigned int _points;
};

class serpent : public enmouvement{
public:
    serpent(position const & p, direction dir, unsigned int j);
    virtual typecase type() const override;
    void remplir_affichage(affichage & a) const;
    void changer_dir(direction d) { modifier_direction(d); };
    std::list<position> const& queue() const { return _queue; };
    void appliquer_deplacement() override;
    void manger_bonus(bonus const&b) { _longueurmax += b.points();};
private:
    std::size_t _longueurmax;
    unsigned int _joueur;
    std::list<position> _queue;
};

class obstacle : public element{
public:
    obstacle(position const & p);
    typecase type() const override { return typecase::obstacle; };
};

enum class etatjeu{
    en_cours,
    gagne,
    perdu
};

class jeu{
public:
    jeu(coord largeur, coord hauteur);

private:
    direction direction_aleatoire() const;
    bool enmouvement_dehors(enmouvement const & e) const;
    bool serpent_perdu(serpent const & s) const;
    std::size_t serpent_mange(serpent const & s);
    std::size_t bonus_supprime_sortants();

public:
    coord largeur() const { return _largeur; }
    coord hauteur() const { return _hauteur; }
    position position_aleatoire() const;
    etatjeu tour_de_jeu(bool deplacerbonus);
    void bonus_ajoute(std::size_t nb);
    void obstacle_ajoute(position const & p);
    void serpent_direction(unsigned int joueur, direction d);
    void affichage_remplir(affichage & a) const;

private:
    coord _largeur, _hauteur;
    std::vector<obstacle> _obstacles;
    std::vector<serpent> _serpents;
    std::vector<bonus> _bonus;
};















