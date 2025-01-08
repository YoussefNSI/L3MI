#pragma once

#include "horaire.hh"
#include <list>
#include <string>
#include <ostream>

using personne = std::string;
using idmessage = unsigned int;

class message {
	public:
	message(personne const& exp, std::list<personne> const& dest);

	idmessage id() const { return _id; }
	personne const& expediteur() const { return _expediteur; }
	std::list<personne> const& destinataires() const { return _destinataires; }

	virtual bool suite_a(idmessage prop) const;
	virtual void sortie_flux(std::ostream& os) const;
	virtual bool heure_correcte() const = 0;
	virtual bool concerne_date(unsigned short a, unsigned short m, unsigned short j) const = 0;
	bool contient_destinataire(personne const& p) const;

	protected:
	virtual std::string type_message() const = 0;

	private:
	idmessage _id;
	personne _expediteur;
	std::list<personne> _destinataires;
	static idmessage _compteur;
};

class proposition : public message {
	public:
	proposition(personne const& exp, std::list<personne> const& dest, std::list<horaire> const& possibles);
	bool heure_correcte() const override;
	bool concerne_date(unsigned short a, unsigned short m, unsigned short j) const override;
	bool contient(horaire const& h) const;

	protected:
	std::string type_message() const override { return "PROPOSITION"; }

	private:
	std::list<horaire> _possibles;
};

class suite : public message {
	public:
	suite(personne const& exp, std::list<personne> const& dest, idmessage s);
	bool suite_a(idmessage prop) const override { return prop == _suite; }
	void sortie_flux(std::ostream& os) const override;
	idmessage suite_de() const { return _suite; }
	private:
	idmessage _suite;
};

class reponse : public suite {
	public:
	reponse(personne const& exp, std::list<personne> const& dest, idmessage s, horaire const& pr);
	void sortie_flux(std::ostream& os) const override;
	bool heure_correcte() const override;
	bool concerne_date(unsigned short a, unsigned short m, unsigned short j) const override;
	horaire const & preference() const { return _preference; }
	protected:
	std::string type_message() const override { return "RÉPONSE"; }
	private:
	horaire _preference;
};

class resultat : public suite {
	public:
	resultat(personne const& exp, std::list<personne> const& dest, idmessage s, horaire const& choix, std::string const& salle);
	void sortie_flux(std::ostream& os) const override;
	bool heure_correcte() const override;
	bool concerne_date(unsigned short a, unsigned short m, unsigned short j) const override;
	horaire const & horaire_choisi() const { return _choix; }
	protected:
	std::string type_message() const override { return "RÉSULTAT"; }
	private:
	horaire _choix;
	std::string _salle;

};
