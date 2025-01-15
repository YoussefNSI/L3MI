#include "entites.hh"

identifiant entite::_compteur(0);

entite::entite()
	: _id(_compteur++) {
}

void entite::sortie_flux(std::ostream& os) const {
	os << _id << " " << symbole();
}

std::ostream& operator<<(std::ostream& os, entite const& e) {
	e.sortie_flux(os);
	return os;
}

coord bombe::_decompte_defaut(10);
bombe::bombe(coord largeur)
	: entite()
	, _decompte(_decompte_defaut)
	, _largeur(largeur)
	, _etat(bombe_etat::decompte) { }

char bombe::symbole() const {
	if (_etat == bombe_etat::decompte)
		return 'B';
	else
		return 'E';
}

void bombe::sortie_flux(std::ostream& os) const {
	entite::sortie_flux(os);
	os << " " << _largeur << " " << _decompte;
}

bool bombe::accepte_joueur() const {
	return false;
}

bool bombe::etat_suivant() {
	if (_decompte == 0)
		if (_etat == bombe_etat::explosion)
			return true;
		else
			exploser();
	else
		--_decompte;
	return false;
}

void bombe::exploser() {
	_etat = bombe_etat::explosion;
	_decompte = 2 * _largeur;
}

void bombe::modifier_decompte(coord nd) {
	_decompte_defaut = nd;
}

char obstacle::symbole() const {
	return '#';
}

bool obstacle::accepte_joueur() const {
	return false;
}

char bonus::symbole() const {
	switch (_type) {
		case bonus_type::amelioration_portee:
			return 'P';
		case bonus_type::amelioration_nombre:
			return 'N';
		case bonus_type::vie_supplementaire:
			return 'V';
	}
	return ' ';
}

void bonus::sortie_flux(std::ostream& os) const {
	entite::sortie_flux(os);
	os << " ";
	switch (_type) {
		case bonus_type::amelioration_portee:
			os << "Amélioration portée";
			break;
		case bonus_type::amelioration_nombre:
			os << "Amélioration nombre";
			break;
		case bonus_type::vie_supplementaire:
			os << "Vie supplémentaire";
			break;
	}
}

bool bonus::accepte_joueur() const {
	return true;
}
