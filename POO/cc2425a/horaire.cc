#include "horaire.hh"
#include <sstream>

std::string horaire::to_string() const {
	std::ostringstream ost;
	ost << _jour << "/" << _mois << "/" << _annee << "-" << _heure << "h";
	return ost.str();
}

bool horaire::concerne_amj(unsigned short a, unsigned short m, unsigned short j) const {
	// Cette méthode n'est pas explicitement demandée, mais elle simplifie l'écriture de la méthode message::concerne_date.
	return (a == _annee) && (m == _mois) && (j == _jour);
}

bool horaire::est_avant(const horaire& h1, const horaire& h2) {
	// Cette méthode n'est pas explicitement demandée, mais elle simplifie base_messages::afficher_apres/
	return (h1._annee < h2._annee) || ((h1._annee == h2._annee) && (h1._mois < h2._mois)) || ((h1._annee == h2._annee) && (h1._mois == h2._mois) && (h1._jour < h2._jour)) || ((h1._annee == h2._annee) && (h1._mois == h2._mois) && (h1._jour == h2._jour) && (h1._heure < h2._heure));
}

bool horaire::est_egal(const horaire& h1, const horaire& h2) {
	// Cette méthode n'est pas explicitement demandée, mais elle simplifie base_messages::repondre.
	return (h1._annee == h2._annee) && (h1._mois == h2._mois) && (h1._jour == h2._jour) && (h1._heure == h2._heure);
}
