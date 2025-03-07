#include "sequence_2.hh"
#include <iostream>

void sequence::ajouter(couleur c) {
	_contenu.push_back(c);
}

indicesequence sequence::taille() const {
	return _contenu.size();
}

void sequence::afficher(couleur c) const {
	afficher(c, std::cout);
}

void sequence::afficher(couleur c, std::ostream & os) const {
	switch (c) {
		case couleur::rouge: os << "rouge"; break;
		case couleur::bleu: os << "bleu"; break;
		case couleur::jaune: os << "jaune"; break;
		case couleur::vert: os << "vert"; break;
	}
}

std::ostream & operator<<(std::ostream & os, couleur c) {
	switch (c) {
		case couleur::rouge: os << "rouge"; break;
		case couleur::bleu: os << "bleu"; break;
		case couleur::jaune: os << "jaune"; break;
		case couleur::vert: os << "vert"; break;
	}
	return os;
}

std::ostream & operator<<(std::ostream & os, sequence const & s) {
	for (indicesequence i(0); i < s.taille(); ++i) {
		os << s.acces(i) << " ";
	}
	return os;
}

couleur sequence::acces(indicesequence i) const {
	return _contenu.at(i);
}

void sequence::vider() {
	_contenu.clear();
}

void sequence::afficher(std::ostream & os) const {
	for (indicesequence i(0); i<_contenu.size(); ++i) {
		afficher(acces(i), os);
		os << " ";
	}
}

bool sequence::comparer(sequence const & s) const {
	return _contenu == s._contenu;
}

void sequence::copier(sequence const & s) {
	_contenu = s._contenu;
}
