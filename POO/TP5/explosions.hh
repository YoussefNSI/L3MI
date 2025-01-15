#pragma once

#include "position.hh"
#include <vector>

using explosions_type = std::uint8_t;
const explosions_type et_haut = 1;
const explosions_type et_droite = 2;
const explosions_type et_bas = 4;
const explosions_type et_gauche = 8;
const explosions_type et_simple = 16;
class explosions {
	public:
	explosions(position const& t);
	void vider();
	void ajouter(position const& p, explosions_type et);
	bool contient_explosion(position const& p) const;
	explosions_type acces(position const& p) const;

	private:
	std::size_t position_vers_indice(position const& p) const;

	private:
	std::vector<std::uint8_t> _expls;
	position _taille;
};
