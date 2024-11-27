#include "sequence.hh"
#include <iostream>

void sc_initialiservide(sequence & s) {
	s.taille = 0;
}

void sc_ajouter(sequence & s, couleur c) {
	s.contenu.at(s.taille++) = c;
}

void sc_copier(sequence & destination, sequence const & source) {
	for (indicesequence i=0; i<source.taille; ++i)
		destination.contenu[i] = source.contenu[i];
	destination.taille = source.taille;
}

couleur sc_acces(sequence const & s, indicesequence i) {
	return s.contenu.at(i);
}

void sc_afficher(couleur c) {
	switch (c) {
		case couleur::rouge: std::cout << "rouge"; break;
		case couleur::bleu: std::cout << "bleu"; break;
		case couleur::jaune: std::cout << "jaune"; break;
		case couleur::vert: std::cout << "vert"; break;
	}
}

void sc_vider(sequence & s) {
	s.taille = 0;
}

void sc_afficher(sequence const & s) {
	for (indicesequence i=0; i<s.taille; ++i) {
		sc_afficher(sc_acces(s,i));
		std::cout << " ";
	}
}

bool sc_comparer(sequence const & a, sequence const & b) {
	if (a.taille != b.taille)
		return false;
	else {
		for (indicesequence i=0; i<a.taille; ++i)
			if (a.contenu[i] != b.contenu[i])
				return false;
		return true;
	}
} 
