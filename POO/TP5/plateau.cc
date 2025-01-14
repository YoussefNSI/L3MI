#include "plateau.hh"

plateau::plateau(position const & t)
	:_entites(t.x() * t.y()),
	_expls(t),
	_taille(t) {
}

bool plateau::ajouter(position const& p, std::unique_ptr<entite> e) {
	if (position_valide(p) && !_entites[position_vers_indice(p)]) {
		_entites[position_vers_indice(p)] = std::move(e);
		return true;
	}
	else
		return false;
}

const std::unique_ptr<entite>& plateau::acces(const position& p) const {
	return _entites[position_vers_indice(p)];
}

void plateau::sortie_graphique(std::ostream& os) const {
	for (coord y = 0; y < _taille.y(); ++y) {
		for (coord x = 0; x < _taille.x(); ++x) {
			auto const& i(_entites[position_vers_indice(position(x, y))]);
			if (i)
				os << i->symbole();
			else if (_expls.contient_explosion(position(x, y)))
				os << "*";
			else
				os << " ";
		}
		os << "\n";
	}
}

void plateau::etat_suivant() {
	_expls.vider();
	for (coord y = 0; y < _taille.y(); ++y) {
		for (coord x = 0; x < _taille.x(); ++x) {
			if (_entites[position_vers_indice(position(x, y))]) {
				auto e_bombe(dynamic_cast<bombe*>(_entites[position_vers_indice(position(x, y))].get()));
				if (e_bombe) {
					if (e_bombe->etat_suivant())
						_entites[position_vers_indice(position(x, y))].reset();
					else if (e_bombe->etat() == bombe_etat::explosion) {
						explosion(position(x, y), std::max(0, e_bombe->largeur() - e_bombe->rebours()), std::min<coord>(e_bombe->largeur(), (e_bombe->largeur() * 2) - e_bombe->rebours()));
					}
				}
			}
		}
	}
	for (coord y = 0; y < _taille.y(); ++y) {
		for (coord x = 0; x < _taille.x(); ++x) {
			if (_expls.contient_explosion(position(x, y))) {
				if (_entites[position_vers_indice(position(x, y))]) {
					auto e_bombe(dynamic_cast<bombe*>(_entites[position_vers_indice(position(x, y))].get()));
					if (e_bombe && (e_bombe->etat() == bombe_etat::decompte))
						e_bombe->exploser();
					if (dynamic_cast<bonus*>(_entites[position_vers_indice(position(x, y))].get()))
						_entites[position_vers_indice(position(x, y))].reset();
				}
			}
		}
	}
}

bool plateau::libre(const position& p) const {
	return position_valide(p) && !_entites[position_vers_indice(p)];
}

bool plateau::accepte_joueur(const position& p) const {
	return position_valide(p) && (!_entites[position_vers_indice(p)] || _entites[position_vers_indice(p)]->accepte_joueur());
}

std::size_t plateau::position_vers_indice(const position& p) const {
	return p.x() + (p.y() * _taille.x());
}

bool plateau::position_valide(position const & p) const {
	return (p.x() >= 0) && (p.y() >= 0) && (p.x() < _taille.x()) && (p.y() < _taille.y());
}

void plateau::explosion(position const& p, coord debut, coord fin) {
	explosion_une_direction(p, 1, 0, debut, fin, et_droite, et_gauche);
	explosion_une_direction(p, -1, 0, debut, fin, et_gauche, et_droite);
	explosion_une_direction(p, 0, 1, debut, fin, et_bas, et_haut);
	explosion_une_direction(p, 0, -1, debut, fin, et_haut, et_bas);
}

void plateau::explosion_une_direction(const position& p, coord dx, coord dy, coord debut, coord fin, explosions_type etdebut, explosions_type etfin) {
	for (coord l(0); l <= fin; ++l) {
		position p2(p.x() + static_cast<coord>(dx) * l, p.y() + static_cast<coord>(dy) * l);
		if (position_valide(p2)) {
			if (_entites[position_vers_indice(p2)]) {
				auto obs(dynamic_cast<obstacle const*>(_entites[position_vers_indice(p2)].get()));
				if (obs)
					break;
			}
			if (l == debut) {
				if (debut == fin)
					_expls.ajouter(p2, et_simple);
				else
					_expls.ajouter(p2, etdebut);
			}
			else if ((l > debut) && (l < fin))
				_expls.ajouter(p2, etdebut | etfin);
			else if (l == fin)
				_expls.ajouter(p2, etfin);
		}
	}
}

std::ostream& operator<<(std::ostream& os, plateau const& p) {
	p.sortie_graphique(os);
	return os;
}
