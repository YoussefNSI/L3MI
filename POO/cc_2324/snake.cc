#include "snake.hh"
#include <cstdlib>

std::string position::tostring() const {
	return "X" + std::to_string(_x) + "Y" + std::to_string(_y);
}

bool position::adjacent(const position& p2) const {
	auto dx(static_cast<signed int>(p2._x) - static_cast<signed int>(_x));
	auto dy(static_cast<signed int>(p2._y) - static_cast<signed int>(_y));
	return ((std::abs(dx) == 1) && (dy == 0)) || ((std::abs(dy) == 1) && (dx == 0));
}

bool position::egale(const position& p2) const {
	return (_x == p2._x) && (_y == p2._y);
}

void affichage::vider() {
	for (auto& i : _cases)
		i = typecase::vide;
}

void affichage::afficher(std::ostream& os) const {
	for (coord y(0); y < _hauteur; ++y) {
		for (coord x(0); x < _largeur; ++x) {
			switch (acces(position(x, y))) {
				case typecase::serpent1:
					os << '1';
					break;
				case typecase::serpent2:
					os << '2';
					break;
				case typecase::obstacle:
					os << '#';
					break;
				case typecase::bonus:
					os << 'B';
					break;
				case typecase::vide:
					os << ' ';
					break;
			}
		}
		os << "\n";
	}

}

element::element(const position& p)
	: _pos(p) {
}

void element::remplir_affichage(affichage & a) const {
	a.fixer(pos(), mon_type());
}

enmouvement::enmouvement(const position& p, direction d)
	: element(p)
	, _dir(d) {
}

void enmouvement::appliquer_deplacement() {
	switch (_dir) {
		case direction::haut:
			modifier_position(position(pos().x(), pos().y()-1));
			break;
		case direction::bas:
			modifier_position(position(pos().x(), pos().y()+1));
			break;
		case direction::droite:
			modifier_position(position(pos().x()+1, pos().y()));
			break;
		case direction::gauche:
			modifier_position(position(pos().x()-1, pos().y()));
			break;
	}
}

bonus::bonus(const position & p, direction dir, unsigned int points)
	:enmouvement(p, dir), _points(points)
{

}

serpent::serpent(const position & p, direction dir, unsigned int j)
	:enmouvement(p, dir), _queue(), _longueurmax(1), _joueur(j) {
}

typecase serpent::mon_type() const {
	if (_joueur == 1)
		return typecase::serpent1;
	else
		return typecase::serpent2;
}

void serpent::remplir_affichage(affichage& a) const {
	enmouvement::remplir_affichage(a);
	for (auto const& i : _queue)
		a.fixer(i, mon_type());
}

void serpent::appliquer_deplacement() {
	_queue.push_back(pos());
	if (_queue.size() > _longueurmax)
		_queue.pop_front();
	enmouvement::appliquer_deplacement();
}

obstacle::obstacle(const position & p)
	:element(p){
}

jeu::jeu(coord largeur, coord hauteur)
	: _largeur(largeur)
	, _hauteur(hauteur) {
	position milieu(_largeur / 2, _hauteur / 2);
	_serpents.push_back(serpent(position(milieu.x() - _largeur/4, milieu.y()), direction::haut, 1));
	_serpents.push_back(serpent(position(milieu.x() + _largeur/4, milieu.y()), direction::bas, 2));
}

direction jeu::direction_aleatoire() const {
	switch (rand() % 4) {
		case 0:
			return direction::haut;
		case 1:
			return direction::bas;
		case 2:
			return direction::droite;
		case 3:
		default:
			return direction::gauche;
	}
}

bool jeu::enmouvement_dehors(const enmouvement& e) const {
	return (e.pos().x() == 0) || (e.pos().y() == 0) || (e.pos().x() == (_largeur - 1)) || (e.pos().y() == (_hauteur - 1));
}

bool jeu::serpent_perdu(const serpent& s) const {
	if (enmouvement_dehors(s))
		return true;
	for (auto const& i : _obstacles)
		if (s.pos().egale(i.pos()))
			return true;
	for (auto const& i : _serpents)
		for (auto const& j : i.queue())
			if (s.pos().egale(j))
				return true;
	return false;
}

std::size_t jeu::serpent_mange(serpent& s) {
	std::size_t compteur(0);
	auto i(_bonus.begin());
	while (i != _bonus.end()) {
		if (s.pos().adjacent(i->pos())) {
			s.manger_bonus(*i);
			i = _bonus.erase(i);
			++compteur;
		}
		else
			++i;
	}
	return compteur;
}

std::size_t jeu::bonus_supprime_sortants() {
	std::size_t compteur(0);
	auto i(_bonus.begin());
	while (i != _bonus.end()) {
		if (enmouvement_dehors(*i)) {
			++compteur;
			i = _bonus.erase(i);
		}
		else
			++i;
	}
	return compteur;
}

position jeu::position_aleatoire() const {
	return position((static_cast<coord>(rand()) % (_largeur - 2)) + 1, (static_cast<coord>(rand()) % (_hauteur - 2)) + 1);
}

etatjeu jeu::tour_de_jeu(bool deplacerbonus) {
	std::size_t bonus_supprimes(0);
	for (auto& i : _serpents) {
		i.appliquer_deplacement();
		bonus_supprimes += serpent_mange(i);
	}
	if (deplacerbonus) {
		for (auto& i : _bonus)
			i.appliquer_deplacement();
		bonus_supprimes += bonus_supprime_sortants();
	}
	bonus_ajoute(bonus_supprimes);
	if (serpent_perdu(_serpents.front())) {
		if (serpent_perdu(_serpents.back()))
			return etatjeu::aucunvainqueur;
		else
			return etatjeu::vainqueur2;
	}
	else if (serpent_perdu(_serpents.back()))
		return etatjeu::vainqueur1;
	return etatjeu::encours;
}

void jeu::bonus_ajoute(std::size_t nb) {
	for (std::size_t i(0); i < nb; ++i) {
		_bonus.push_back(bonus(position_aleatoire(), direction_aleatoire(), static_cast<unsigned int>(rand()) % 10 + 5));
	}
}

void jeu::obstacle_ajoute(position const& p) {
	_obstacles.push_back(obstacle(p));
}

void jeu::serpent_direction(unsigned int joueur, direction d) {
	_serpents.at(joueur).changer_dir(d);
}

void jeu::affichage_remplir(affichage& aff) const {
	aff.vider();
	for (auto const& i : _obstacles)
		i.remplir_affichage(aff);
	for (auto const& i : _bonus)
		i.remplir_affichage(aff);
	for (auto const& i : _serpents)
		i.remplir_affichage(aff);
}





