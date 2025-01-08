#pragma once

#include "message.hh"
#include <list>

class base_messages {
	public:
	base_messages() = default;

	void ajouter(proposition const& p) { _prop.push_back(p); }
	void ajouter(reponse const& p) { _rep.push_back(p); }
	void ajouter(resultat const& p) { _res.push_back(p); }

	void afficher_apres(horaire const& h) const;
	bool repondre(idmessage p, personne const& d, horaire const& h);
	bool verifier_reponses() const;
	void purger();
	void supprimer(idmessage p);

	private:
	std::list<proposition>::const_iterator rechercher_prop(idmessage i) const;
	bool contient_resultat(idmessage i) const;
	private:
	std::list<proposition> _prop;
	std::list<reponse> _rep;
	std::list<resultat> _res;
};
