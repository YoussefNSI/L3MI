#pragma once

#include "explosions.hh"
#include "entites.hh"
#include <memory>

class plateau {
	public:
	plateau(position const& t);
	plateau(plateau const& p) =delete;
	plateau& operator=(plateau const& p) =delete;

	bool ajouter(position const & p, std::unique_ptr<entite> e);
	std::unique_ptr<entite> const& acces(position const& p) const;
	explosions const& acces_explosions() const { return _expls; };
	void sortie_graphique(std::ostream& os) const;
	void etat_suivant();
	position const & taille() const { return _taille; }
	bool libre(position const & p) const;
	bool accepte_joueur(position const& p) const;

	static const unsigned int bloc_w = 64;
	static const unsigned int bloc_h = 48;

	private:
	std::size_t position_vers_indice(position const& p) const;
	bool position_valide(position const& p) const;
	void explosion(position const& p, coord debut, coord fin);
	void explosion_une_direction(position const& p, coord dx, coord dy, coord debut, coord fin, explosions_type etdebut, explosions_type etfin);

	private:
	std::vector<std::unique_ptr<entite>> _entites;
	explosions _expls;
	position _taille;
};
std::ostream& operator<<(std::ostream& os, plateau const& p);
