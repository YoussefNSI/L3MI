#pragma once
#include <vector>
#include <iostream>

enum class couleur {
	rouge,
	bleu,
	jaune,
	vert
};

std::ostream & operator<<(std::ostream & os, couleur c);

class sequence {
	public:
		using indice = std::size_t;

		sequence() =default;
		sequence(sequence const & s) =default;
		~sequence() =default;
		sequence & operator=(sequence const & s) =default;

		void ajouter(couleur c) {
			_couleurs.push_back(c); }
		indice taille() const {
			return _couleurs.size(); }
		void vider() {
			_couleurs.clear(); }
		couleur operator[](indice i) const {
			return _couleurs.at(i); }
		bool operator==(sequence const & s) const {
			return _couleurs == s._couleurs; }

	private:
		std::vector<couleur> _couleurs;
		friend std::ostream & operator<<(std::ostream & os, sequence const & s);
};
