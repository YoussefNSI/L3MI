#include "grille.hh"
#include <string>
#include <iostream>

int main() {
	std::vector<grille> grilles(2,grille(20,20));
	unsigned int actuel=0;
	grilles[0].ajouterstructure(structure::floraison, 10, 10);
	grilles[0].afficher();
	while (true) {
		std::string s; std::getline(std::cin, s);
		grilles[actuel].evolution(grilles[1-actuel]);
		actuel = 1-actuel;
		grilles[actuel].afficher();
	};
	return 0;
}
