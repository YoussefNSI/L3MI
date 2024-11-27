#pragma once
#include <cstddef>

enum class couleur {
	rouge,
	bleu,
	jaune,
	vert
};

using indicesequence = std::size_t;
struct sequence {
	couleur * contenu;
	indicesequence taille;
	indicesequence reserve;
};

void sc_initialiservide(sequence & s);
void sc_ajouter(sequence & s, couleur c);
void sc_copier(sequence & destination, sequence const & source);
couleur sc_acces(sequence const & s, indicesequence i);
void sc_afficher(couleur c);
void sc_vider(sequence & s);
void sc_afficher(sequence const & s);
bool sc_comparer(sequence const & a, sequence const &b);
void sc_detruire(sequence & s);
