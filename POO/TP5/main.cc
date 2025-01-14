#include "echiquier.hh"
#include <iostream>

int main() {
	echiquier e;
	e.depart();
	/*	Test Sauvegarde / chargement.
	{	std::ofstream ofs("toto");
		e.sauvegarde(ofs);
	}
	echiquier e2;
	{	std::ifstream ifs("toto");
		e2.chargement(ifs);
	}
	e2.affichage(std::cout);
	*/
	couleur joueur(couleur::blanc);
	while (!e.aperdu(joueur)) {
		e.affichage(std::cout);
		std::cout << ((joueur==couleur::blanc) ? "blanc" : "noir") << " pièce à déplacer ?";
		coord x, y;
		std::cin >> x >> y;
		if (e.contientpiececouleur(position(x,y), joueur)) {
			std::cout << ((joueur==couleur::blanc) ? "blanc" : "noir") << " case destination ?";
			coord x2, y2;
			std::cin >> x2 >> y2;
			auto ok = e.deplacer(position(x,y), position(x2,y2));
			if (ok)
				if (joueur == couleur::noir)
					joueur=couleur::blanc;
				else
					joueur=couleur::noir;
			else
				std::cout << "déplacement impossible\n";
		}
	}
	std::cout << ((joueur==couleur::blanc) ? "blanc" : "noir") << " a perdu\n";
	return 0;
}
