#include "explosions.hh"

explosions::explosions(const position& t)
	: _expls(t.x() * t.y(), 0)
	, _taille(t) {
}

void explosions::vider() {
	for (auto& i : _expls)
		i = 0;
}

void explosions::ajouter(const position& p, explosions_type et) {
	_expls.at(position_vers_indice(p)) |= et;
}

bool explosions::contient_explosion(const position& p) const {
	return _expls.at(position_vers_indice(p)) != 0;
}

explosions_type explosions::acces(const position& p) const {
	return _expls.at(position_vers_indice(p));
}

std::size_t explosions::position_vers_indice(const position& p) const {
	return p.x() + (p.y() * _taille.x());
}
