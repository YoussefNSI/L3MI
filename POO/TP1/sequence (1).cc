#include "sequence.hh"
#include <iostream>

void sc_initialiservide(sequence & s) {
	s.contenu = nullptr;
	s.taille = 0;
}

void sc_ajouter(sequence & s, couleur c) {
	couleur * nouveau(new couleur[s.taille+1]);
	for (indicesequence i(0); i<s.taille; ++i)
		nouveau[i] = s.contenu[i];
	nouveau[s.taille] = c;
	delete [] s.contenu;
	s.contenu = nouveau;
	++s.taille;
}

void sc_copier(sequence & destination, sequence const & source) {
	if (destination.taille != source.taille) {
		delete [] destination.contenu;
		if (source.taille == 0)
			destination.contenu = nullptr;
		else
			destination.contenu = new couleur[source.taille];
		destination.taille = source.taille;
	}
	for (indicesequence i(0); i<source.taille; ++i)
		destination.contenu[i] = source.contenu[i];
}

couleur sc_acces(sequence const & s, indicesequence i) {
	return s.contenu[i];
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
	delete []s.contenu;
	s.contenu = nullptr;
}	

void sc_afficher(sequence const & s) {
	for (indicesequence i=0; i<s.taille; ++i) {
		sc_afficher(sc_acces(s, i));
		std::cout << " ";
	}
}

bool sc_comparer(sequence const & a, sequence const & b) {
	if (a.taille != b.taille)
		return false;
	else {
		for (indicesequence i=0; i<a.taille; ++i)
			if (sc_acces(a,i) != sc_acces(b,i))
				return false;
		return true;
	}
}

void sc_detruire(sequence & s) {
	delete [] s.contenu;
}
