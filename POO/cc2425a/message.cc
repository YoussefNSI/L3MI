#include "message.hh"

idmessage message::_compteur(0);

message::message(const personne& exp, const std::list<personne>& dest)
	: _id(_compteur++)
	, _expediteur(exp)
	, _destinataires(dest) {
}

bool message::suite_a(idmessage prop) const {
	return false;
}

void message::sortie_flux(std::ostream& os) const {
	os << _id << " - " << type_message() << " " << _expediteur << "->";
	for (auto const& i : _destinataires)
		os << i << " ";
}

bool message::contient_destinataire(const personne& p) const {
	// Cette méthode n'est pas demandée explicitement, mais elle simplifie base_messages::repondre.
	for (auto const& i : _destinataires)
		if (i == p)
			return true;
	return false;
}

proposition::proposition(const personne& exp, const std::list<personne>& dest, const std::list<horaire>& possibles)
	: message(exp, dest)
	, _possibles(possibles) {
}

bool proposition::heure_correcte() const {
	for (auto const& i : _possibles)
		if ((i.heure() < 8) || (i.heure() > 18))
			return false;
	return true;
}

bool proposition::concerne_date(unsigned short a, unsigned short m, unsigned short j) const {
	for (auto const& i : _possibles)
		if (i.concerne_amj(a, m, j))
			return true;
	return false;
}

bool proposition::contient(const horaire& h) const {
	// Méthode pas demandée explicitement, mais qui simplifie l'écriture de base_messages::repondre.
	for (auto const& i : _possibles)
		if (horaire::est_egal(h, i))
			return true;
	return false;
}

suite::suite(const personne& exp, const std::list<personne>& dest, idmessage s)
	: message(exp, dest)
	, _suite(s) {
}

void suite::sortie_flux(std::ostream& os) const {
	message::sortie_flux(os);
	os << "Proposition " << _suite;
}

reponse::reponse(const personne& exp, const std::list<personne>& dest, idmessage s, const horaire& pr)
	: suite(exp, dest, s)
	, _preference(pr) {
}

void reponse::sortie_flux(std::ostream& os) const {
	suite::sortie_flux(os);
	os << " " << _preference.to_string();
}

bool reponse::heure_correcte() const {
	return (_preference.heure() >= 8) && (_preference.heure() <= 18);
}

bool reponse::concerne_date(unsigned short a, unsigned short m, unsigned short j) const {
	return _preference.concerne_amj(a, m, j);
}

resultat::resultat(const personne& exp, const std::list<personne>& dest, idmessage s, const horaire& choix, const std::string& salle)
	: suite(exp, dest, s)
	, _choix(choix)
	, _salle(salle) {
}

void resultat::sortie_flux(std::ostream& os) const {
	suite::sortie_flux(os);
	os << " Réunion fixée le " << _choix.to_string() << " en " << _salle;
}

bool resultat::heure_correcte() const {
	return (_choix.heure() >= 8) && (_choix.heure() <= 18);
}

bool resultat::concerne_date(unsigned short a, unsigned short m, unsigned short j) const {
	return _choix.concerne_amj(a, m, j);
}
