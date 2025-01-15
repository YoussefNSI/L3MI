#pragma once

#include "position.hh"
#include "plateau.hh"

enum class direction {
	haut,
	bas,
	droite,
	gauche,
	stop
};

class mobile {
	public:
	mobile(position const& p);

	position const& pos() const { return _pos; }
	unsigned short x() const { return _pos.x() * plateau::bloc_w + _dx; }
	unsigned short y() const { return _pos.y() * plateau::bloc_h + _dy; }
	signed short dx() const { return _dx; }
	signed short dy() const { return _dy; }
	bool au_centre() const { return (_dx == 0) && (_dy == 0); }

	direction dir_voulue() const { return _dir_voulue; }
	direction dir_actuelle() const { return _dir_actuelle; }

	void fixer_dir_voulue(direction d) { _dir_voulue = d; }

	virtual void etat_suivant(plateau const & p);

	virtual bool craint_explosion() const = 0;
	virtual bool subit_explosion() = 0;

	static direction direction_aleatoire();

	private:
	void appliquer_deplacement(plateau const& p);
	void mise_a_jour_dir_actuelle();

	private:
	position _pos;

	signed short _dx;
	signed short _dy;

	direction _dir_actuelle;
	direction _dir_voulue;
};

class joueur : public mobile {
	public:
	joueur(position const & p, unsigned short num);

	unsigned short numero() const { return _numero; }
	unsigned short nb_bombes() const { return _nb_bombes; }
	unsigned short portee_bombes() const { return _portee_bombes; }
	unsigned short nb_vies() const { return _nb_vies; }
	unsigned short invincibilite() const { return _invincibilite; }

	void etat_suivant(plateau const & p) override;
	bool craint_explosion() const override;
	bool subit_explosion() override;

	private:
	unsigned short _numero;
	unsigned short _nb_bombes;
	unsigned short _portee_bombes;
	unsigned short _nb_vies;
	unsigned short _invincibilite;
};

class ennemi: public mobile {
	public:
	ennemi(position const & p);

	bool craint_explosion() const override;
	bool subit_explosion() override;
};
