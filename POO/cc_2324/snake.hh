#pragma once
#include <list>
#include <vector>
#include <string>
#include <iostream>

using coord = unsigned short;

class position {
	public:
	position(coord x, coord y)
		: _x(x)
		, _y(y) { }
	coord x() const { return _x; }
	coord y() const { return _y; }
	std::string tostring() const;
	bool adjacent(position const& p2) const;
	bool egale(position const& p2) const;

	private:
	coord _x;
	coord _y;
};

enum class typecase {
	serpent1,
	serpent2,
	obstacle,
	bonus,
	vide,
};

class affichage {
	public:
	affichage(coord largeur, coord hauteur)
		: _cases(largeur * hauteur, typecase::vide)
		, _largeur(largeur)
		, _hauteur(hauteur) { }
	void vider();
	typecase acces(position const & p) const { return _cases.at(p.x() + p.y() * _largeur); }
	void fixer(position const & p, typecase c) { _cases.at(p.x() + p.y() * _largeur) = c; }
	void afficher(std::ostream& os) const;
	private:
	std::vector<typecase> _cases;
	coord _largeur;
	coord _hauteur;
};

class element {
	public:
	element(position const& p);
	virtual ~element() = default;
	position const& pos() const { return _pos; }
	virtual typecase mon_type() const = 0;
	virtual void remplir_affichage(affichage & a) const;

	protected:
	void modifier_position(position const& p) { _pos = p; }

	private:
	position _pos;
};

enum class direction {
	haut, bas, droite, gauche
};

class enmouvement : public element {
	public:
	enmouvement(position const& p, direction d);
	direction dir() const { return _dir; }
	virtual void appliquer_deplacement();

	protected:
	void modifier_dir(direction d) { _dir = d; }

	private:
	direction _dir;
};

class bonus : public enmouvement {
	public:
	bonus(position const& p, direction dir, unsigned int points);
	typecase mon_type() const override { return typecase::bonus; }
	unsigned int points() const { return _points; }
	private:
	unsigned int _points;
};

class serpent : public enmouvement {
	public:
	serpent(position const& p, direction dir, unsigned int j);
	typecase mon_type() const override;
	void remplir_affichage(affichage & a) const override;
	void changer_dir(direction d) { modifier_dir(d); }
	std::list<position> const& queue() const { return _queue; }
	void appliquer_deplacement() override;
	void manger_bonus(bonus const& b) { _longueurmax += b.points(); }
	private:
	std::list<position> _queue;
	std::size_t _longueurmax;
	unsigned int _joueur;
};

class obstacle: public element {
	public:
	obstacle(position const& p);
	typecase mon_type() const override { return typecase::obstacle; }
};

enum class etatjeu {
	encours,
	vainqueur1,
	vainqueur2,
	aucunvainqueur,
};

class jeu {
	public:
	jeu(coord largeur, coord hauteur);

	private:
	direction direction_aleatoire() const;
	bool enmouvement_dehors(enmouvement const& e) const;
	bool serpent_perdu(serpent const& s) const;
	std::size_t serpent_mange(serpent& s);
	std::size_t bonus_supprime_sortants();

	public:
	coord largeur() const { return _largeur; }
	coord hauteur() const { return _hauteur; }
	position position_aleatoire() const;
	etatjeu tour_de_jeu(bool deplacerbonus);
	void bonus_ajoute(std::size_t nb);
	void obstacle_ajoute(position const& p);
	void serpent_direction(unsigned int joueur, direction d);
	void affichage_remplir(affichage& aff) const;

	private:
	coord _largeur;
	coord _hauteur;
	std::vector<obstacle> _obstacles;
	std::vector<bonus> _bonus;
	std::vector<serpent> _serpents;
};






