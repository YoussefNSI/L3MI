#pragma once
#include <array>

enum class couleur {
	rouge,
	bleu,
	jaune,
	vert
};

using indicesequence = unsigned int;
struct sequence {
	std::array<couleur, 100> contenu;
	indicesequence taille;
};

void sc_initialiservide(sequence & s);
void sc_ajouter(sequence & s, couleur c);
void sc_copier(sequence & destination, sequence const & source);
couleur sc_acces(sequence const & s, indicesequence i);
void sc_afficher(couleur c);
void sc_vider(sequence & s);
void sc_afficher(sequence const & s);
bool sc_comparer(sequence const & a, sequence const &b);
