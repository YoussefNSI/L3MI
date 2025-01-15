#include "proprietaire.hh"

void proprietaire::sortieflux(std::ostream & os) const {
	os << typeprop() << " Nom: " << nom() << " Adresse: " << adresse();
 }

std::shared_ptr<proprietaire> proprietaire::fabriquepersonne(std::string const & nom, std::string const & adresse, std::string naissance) {
	return std::shared_ptr<personne>(new personne(nom, adresse, naissance));
}

std::shared_ptr<proprietaire> proprietaire::fabriquesociete(std::string const & nom, std::string const & adresse, std::shared_ptr<proprietaire> prop) {
	if (prop->typeprop() == type::personne)
		return std::shared_ptr<societe>(new societe(nom, adresse, std::dynamic_pointer_cast<personne>(prop)));
	else
		return nullptr;
}

std::shared_ptr<proprietaire> proprietaire::fabriquecouple(std::shared_ptr<proprietaire> p1, std::shared_ptr<proprietaire> p2) {
	if ((p1->typeprop() == type::personne) && (p2->typeprop() == type::personne) && (p1->adresse() == p2->adresse()))
		return std::shared_ptr<couple>(new couple(std::dynamic_pointer_cast<personne>(p1), std::dynamic_pointer_cast<personne>(p2)));
	else
		return nullptr;
}

bool proprietaire::lie_a(std::shared_ptr<proprietaire> /*p*/) const {
	return false;
}

std::ostream & operator<<(std::ostream & os, proprietaire::type t) {
	switch (t) {
		case proprietaire::type::personne: os << "personne"; break;
		case proprietaire::type::societe: os << "societe"; break;
		case proprietaire::type::couple: os << "couple"; break;
	}
	return os;
}

std::ostream & operator<<(std::ostream & os, proprietaire const & p) {
	p.sortieflux(os);
	return os;
}

void personne::sortieflux(std::ostream & os) const {
	proprietaire::sortieflux(os);
	os << " Naissance: " << _naissance;
}

void societe::sortieflux(std::ostream & os) const {
	proprietaire::sortieflux(os);
	os << " Gérant: " << (*_gerant);
}

bool societe::lie_a(std::shared_ptr<proprietaire> p) const {
	auto pp(std::dynamic_pointer_cast<personne>(p));
	return pp && (_gerant == pp);
}

bool couple::lie_a(std::shared_ptr<proprietaire> p) const {
	auto pp(std::dynamic_pointer_cast<personne>(p));
	return pp && ((_p1 == p) || (_p2 == p));
}
