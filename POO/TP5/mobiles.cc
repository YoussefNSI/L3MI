#include "mobiles.hh"
#include <cstdlib>

mobile::mobile(position const & pos)
	: _pos(pos), _dx(0), _dy(0)
	, _dir_actuelle(direction::stop), _dir_voulue(direction::stop) {
}

void mobile::etat_suivant(const plateau& p) {
	mise_a_jour_dir_actuelle();
	appliquer_deplacement(p);
}

direction mobile::direction_aleatoire() {
	auto r(rand() % 4);
	if (r == 0)
		return direction::haut;
	else if (r == 1)
		return direction::droite;
	else if (r == 2)
		return direction::bas;
	else
		return direction::gauche;
}

void mobile::appliquer_deplacement(plateau const& p) {
	switch (_dir_actuelle) {
		case direction::haut:
			if ((_dy != 0) || (p.accepte_joueur(position(_pos.x(), _pos.y() - 1)))) {
				_dy-=4;
				if (_dy == -static_cast<signed short>(plateau::bloc_h) / 2) {
					_dy = plateau::bloc_h / 2;
					_pos.sety(_pos.y() - 1);
				}
			}
			break;
		case direction::bas:
			if ((_dy != 0) || (p.accepte_joueur(position(_pos.x(), _pos.y() + 1)))) {
				_dy+=4;
				if (_dy == static_cast<signed short>(plateau::bloc_h) / 2) {
					_dy = -_dy;
					_pos.sety(_pos.y() + 1);
				}
			}
			break;
		case direction::droite:
			if ((_dx != 0) || (p.accepte_joueur(position(_pos.x() + 1, _pos.y())))) {
				_dx+=4;
				if (_dx == static_cast<signed short>(plateau::bloc_w) / 2) {
					_dx = -_dx;
					_pos.setx(_pos.x() + 1);
				}
			}
			break;
		case direction::gauche:
			if ((_dx != 0) || (p.accepte_joueur(position(_pos.x() - 1, _pos.y())))) {
				_dx-=4;
				if (_dx == -static_cast<signed short>(plateau::bloc_w) / 2) {
					_dx = plateau::bloc_w / 2;
					_pos.setx(_pos.x() - 1);
				}
			}
			break;
		case direction::stop:
			break;
	}
}

void mobile::mise_a_jour_dir_actuelle() {
	switch (_dir_voulue) {
		case direction::haut:
		case direction::bas:
			if (_dx == 0)
				_dir_actuelle = _dir_voulue;
			break;
		case direction::droite:
		case direction::gauche:
			if (_dy == 0)
				_dir_actuelle = _dir_voulue;
			break;
		case direction::stop:
			if ((_dx == 0) && (_dy == 0))
				_dir_actuelle = _dir_voulue;
			break;
	}
}

joueur::joueur(position const & p, unsigned short num)
	:mobile(p), _numero(num), _nb_bombes(3), _portee_bombes(3), _nb_vies(3), _invincibilite(0) {
}

void joueur::etat_suivant(const plateau& p) {
	mobile::etat_suivant(p);
	if (_invincibilite > 0)
		--_invincibilite;
}

bool joueur::craint_explosion() const {
	return _invincibilite == 0;
}

bool joueur::subit_explosion() {
	if (_invincibilite == 0) {
		--_nb_vies;
		_invincibilite = 50;
		return _nb_vies == 0;
	}
	return false;
}

ennemi::ennemi(const position& p):mobile(p){
}

bool ennemi::craint_explosion() const {
	return true;
}

bool ennemi::subit_explosion() {
	return true;
}
