#include "breakout.hh"
#include <algorithm>
#include <iostream>

bool objet::detruit_quand_collision() const {
	return false;
}

bool intersectionintervalle(coord a1, coord a2, coord b1, coord b2) {
	return b1 < a2 && a1 < b2;
}

bool objet::collision(const position_t & p1, const objet& o2) {
	return intersectionintervalle(p1.x(), p1.x() + taille().w(), o2.position().x(), o2.position().x() + o2.taille().w()) && intersectionintervalle(p1.y(), p1.y() + taille().h(), o2.position().y(), o2.position().y() + o2.taille().h());
}

void objet::sortie_flux(std::ostream& os) const {
	os << type_objet() << " Pos " << position().x() << " " << position().y() << " Tai " << taille().w() << " " << taille().h();
}

std::ostream& operator<<(std::ostream& os, const objet& o) {
	o.sortie_flux(os);
	return os;
}


mur::mur(const position_t& p, const taille_t& t)
	: objet(p)
	, _taille(t) {
}

std::unique_ptr<objet> mur::clone() const {
	return std::make_unique<mur>(*this);
}

taille_t mur::taille() const {
	return _taille;
}

std::string mur::type_objet() const {
	return "Mur";
}

std::unique_ptr<objet> bloc::clone() const {
	return std::make_unique<bloc>(*this);
}

taille_t bloc::taille() const {
	return taille_t(20, 10);
}

bool bloc::detruit_quand_collision() const {
	return _typebloc == type::destructible;
}

std::string bloc::type_objet() const {
	switch (_typebloc) {
		case destructible:
			return "Bloc destructible";
		case indestructible:
			return "Bloc indestructible";
	}
	return "";
}

position_t objet_en_mouvement::calculer_deplacement() const {
	auto result(position());
	return result += vitesse();
}

void objet_en_mouvement::appliquer_deplacement_simple() {
	modifier_position(calculer_deplacement());
}

void objet_en_mouvement::sortie_flux(std::ostream& os) const {
	objet::sortie_flux(os);
	os << " Vit " << vitesse().dx() << " " << vitesse().dy();
}

std::string balle::type_objet() const {
	return "Balle";
}

std::unique_ptr<objet> balle::clone() const {
	return std::make_unique<balle>(*this);
}

void balle::appliquer_deplacement_collision(objet const& o) {
	modifier_position(calculer_deplacement());
	if (!collision(position() + vitesse().rebond_horizontal(), o))
		_vitesse = _vitesse.rebond_horizontal();
	else if (!collision(position() + vitesse().rebond_vertical(), o))
		_vitesse = _vitesse.rebond_vertical();
	else // Ne devrait jamais arriver normalement, sauf si deux objets se déplacent l'un en direction de l'autre. On repart à l'opposé.
		_vitesse = _vitesse.rebond_horizontal().rebond_vertical();
}

std::string raquette::type_objet() const {
	return "Raquette";
}

std::unique_ptr<objet> raquette::clone() const {
	return std::make_unique<raquette>(*this);
}

vitesse_t raquette::vitesse() const {
	switch (_dir) {
		case direction::droite:
			return vitesse_t(1, 0);
		case direction::gauche:
			return vitesse_t(-1, 0);
		case direction::stop:
			return vitesse_t(0, 0);
	}
	return vitesse_t(0, 0);
}

void raquette::appliquer_deplacement_collision(objet const & /*o*/) {
	// On ne fait rien : la raquette ne bouge pas en cas de collision (aussi bien avec un mur du bord qu'avec la balle).
}

void raquette::modifier_direction(direction d) {
	_dir = d;
}


const char* breakout_exception::what() const noexcept {
	switch (_tp) {
		case type::plusieurs_raquettes:
			return "Il ne doit y avoir qu'une seule raquette.";
		case type::trop_petit:
			return "Le jeu est trop petit.";
		case type::impossible_ici:
			return "Impossible d'ajouter l'objet ici";
		case type::pas_de_balle:
			return "Pas de balle";
	}
	return "";
}



jeu::jeu(coord hauteur)
	: _hauteur(hauteur) {
	if (_hauteur < 200)
		throw breakout_exception(breakout_exception::type::trop_petit);
}

jeu::jeu(const jeu& j)
	: _hauteur(j._hauteur) {
	for (auto const& i : j._objets)
		_objets.push_back(i->clone());
}

jeu& jeu::operator=(const jeu& j) {
	if (this != &j) {
		_objets.clear();
		_hauteur = j._hauteur;
		for (auto const& i : j._objets)
			_objets.push_back(i->clone());
	}
	return *this;
}

void jeu::ajouter(std::unique_ptr<objet> obj) {
	if (dynamic_cast<raquette const*>(obj.get())) {
		if (std::find_if(_objets.begin(), _objets.end(), [](auto const& i) { return dynamic_cast<raquette const*>(i.get()) != nullptr; }) != _objets.end())
			throw breakout_exception(breakout_exception::type::plusieurs_raquettes);
	}
	auto f(std::find_if(_objets.begin(), _objets.end(), [&obj](auto const& i) { return obj->collision(obj->position(), *i); }));
	if (f != _objets.end())
		throw breakout_exception(breakout_exception::type::impossible_ici);
	_objets.push_back(std::move(obj));
}

void jeu::bouger_raquette(raquette::direction d) {
	auto f(std::find_if(_objets.begin(), _objets.end(), [](auto const& i) { return dynamic_cast<raquette const*>(i.get()) != nullptr; }));
	if (f != _objets.end())
		dynamic_cast<raquette&>(**f).modifier_direction(d);
}

balle const& jeu::acces_balle() const {
	auto f(std::find_if(_objets.begin(), _objets.end(), [](auto const& i) { return dynamic_cast<balle const*>(i.get()) != nullptr; }));
	if (f == _objets.end())
		throw breakout_exception(breakout_exception::type::pas_de_balle);
	else
		return dynamic_cast<balle const&>(**f);
}

std::size_t jeu::nombre_blocs_destructibles() const {
	return std::count_if(_objets.begin(), _objets.end(), [](auto const& i) { auto bl(dynamic_cast<bloc const *>(i.get())); return bl && bl->type_bloc() == bloc::type::destructible; });
}

const std::list<std::unique_ptr<objet>>& jeu::objets() const {
	return _objets;
}

jeu::etat jeu::evoluer() {
	auto coll(deplacer());
	effacer_objets(coll);
	if (nombre_blocs_destructibles() == 0)
		return etat::gagne;
	try {
		auto b(acces_balle());
		if (b.position().y() >= _hauteur)
			return etat::perdu;
	} catch (breakout_exception const&) {
		return etat::perdu; // il n'y a plus de balle !
	}
	return etat::en_cours;
}

std::list<std::list<std::unique_ptr<objet>>::iterator> jeu::deplacer() {
	std::list<std::list<std::unique_ptr<objet>>::iterator> resultat;
	for (auto it1(_objets.begin()); it1 != _objets.end(); ++it1) {
		auto* oem(dynamic_cast<objet_en_mouvement*>(it1->get()));
		if (oem) {
			auto nouvpos(oem->calculer_deplacement());
			auto collision_it(std::find_if(_objets.begin(), _objets.end(), [it1, &nouvpos](auto const& i) { return (i != *it1) && (*it1)->collision(nouvpos, *i); }));
			if (collision_it == _objets.end())
				oem->appliquer_deplacement_simple();
			else {
				oem->appliquer_deplacement_collision(**collision_it);
				ensemble_objets_ajouter(resultat, it1);
				ensemble_objets_ajouter(resultat, collision_it);
			}
		}
	}
	return resultat;
}

void jeu::ensemble_objets_ajouter(std::list<std::list<std::unique_ptr<objet>>::iterator>& c, std::list<std::unique_ptr<objet>>::iterator i) {
	if ((*i)->detruit_quand_collision() && (std::find(c.begin(), c.end(), i) == c.end()))
		c.push_back(i);
}

void jeu::effacer_objets(const std::list<std::list<std::unique_ptr<objet>>::iterator>& lc) {
	for (auto i : lc)
		_objets.erase(i);
}

