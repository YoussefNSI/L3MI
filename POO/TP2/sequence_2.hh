#pragma once
#include <vector>
#include <iostream>

enum class couleur {
	rouge,
	bleu,
	jaune,
	vert
};

using indicesequence = std::size_t;

class sequence {
	public:
	sequence() =default;
	sequence(sequence const & s) =default;
	~sequence() =default;
	void ajouter(couleur c);
	indicesequence taille() const;
	void afficher(couleur c) const;
	void afficher(couleur c, std::ostream & os) const;
	couleur acces(indicesequence i) const;
	void vider();
	void afficher(std::ostream & os) const;
	bool comparer(sequence const & s) const;
	void copier(sequence const & s);
	private:
	std::vector<couleur> _contenu;
};

std::ostream & operator<<(std::ostream & os, couleur c);
std::ostream & operator<<(std::ostream & os, sequence const & s);
