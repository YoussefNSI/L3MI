#pragma once
#include "pieces.hh"
#include <vector>
#include <memory>
#include <fstream>

class echiquier {
	public:
		echiquier() =default;
		echiquier(echiquier const & e);
		echiquier & operator=(echiquier const & e);
		void ajout(std::unique_ptr<piece> p) {
			_pieces.push_back(std::move(p));
		}
		unsigned int valeurdujoueur(couleur c) const;
		bool deplacer(position const & src, position const & dst);
		void affichage(std::ostream & os) const;
		void sauvegarde(std::ofstream & os) const;
		void chargement(std::ifstream & os);
		void depart();
		bool aperdu(couleur c) const;
		bool contientpiececouleur(position const & p, couleur c);
	private:
		std::vector<std::unique_ptr<piece>> _pieces;
};
