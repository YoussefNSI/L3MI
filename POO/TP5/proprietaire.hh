#pragma once

#include <string>
#include <memory>
#include <iostream>

// On fait cette déclaration avancée pour pouvoir déclarer la méthode lie_a.
class personne;

class proprietaire {
	public:
		enum class type {
			personne,
			societe,
			couple
		};
		virtual ~proprietaire() =default;
		virtual std::string nom() const =0;
		virtual std::string const & adresse() const =0;
		virtual type typeprop() const =0;	
		std::string etiquetteexpedition() const { return nom() + " " + adresse(); }
		virtual void sortieflux(std::ostream & os) const;

		static std::shared_ptr<proprietaire> fabriquepersonne(std::string const & nom, std::string const & adresse, std::string naissance);
		static std::shared_ptr<proprietaire> fabriquesociete(std::string const & nom, std::string const & adresse, std::shared_ptr<proprietaire> prop);
		static std::shared_ptr<proprietaire> fabriquecouple(std::shared_ptr<proprietaire> p1, std::shared_ptr<proprietaire> p2);

		/* Cette méthode n'est pas explicitement demandée, mais permet d'écrire la méthode comptes de banque de façon élégante, elle a pour but de calculer si le proprietaire "courant" est lié au propriétaire P passé en paramètre, c'est à dire que P est soit le gérant de la société, soit un membre du couple (voir les redéfinitions dans les sous-classes). */
		virtual bool lie_a(std::shared_ptr<proprietaire> p) const;
};

std::ostream & operator<<(std::ostream & os, proprietaire::type t);
std::ostream & operator<<(std::ostream & os, proprietaire const & p);

class personne: public proprietaire {
	private:
		personne(std::string const & nom, std::string const & adresse, std::string const & naissance)
			:_nom(nom), _adresse(adresse), _naissance(naissance) {}
	public:
		std::string nom() const override { return _nom; }
		std::string const & adresse() const override { return _adresse; }
		type typeprop() const override { return type::personne; }	
		void sortieflux(std::ostream & os) const override;
	private:
		std::string _nom;
		std::string _adresse;
		std::string _naissance;
	friend proprietaire;
};

class societe: public proprietaire {
	private:
		societe(std::string const & nom, std::string const & adresse, std::shared_ptr<personne> gerant)
			:_nom(nom), _adresse(adresse), _gerant(gerant) {}
	public:
		std::string nom() const override { return _nom; }
		std::string const & adresse() const override { return _adresse; }
		type typeprop() const override { return type::societe; }	
		void sortieflux(std::ostream & os) const override;
		bool lie_a(std::shared_ptr<proprietaire> p) const override;
	private:
		std::string _nom;
		std::string _adresse;
		std::shared_ptr<personne> _gerant;
	friend proprietaire;
};

class couple: public proprietaire {
	private:
		couple(std::shared_ptr<personne> p1, std::shared_ptr<personne> p2)
			:_p1(p1), _p2(p2) {}
	public:
		std::string nom() const override { return _p1->nom() + " " + _p2->nom(); }
		std::string const & adresse() const override { return _p1->adresse(); }
		type typeprop() const override { return type::couple; }
		bool lie_a(std::shared_ptr<proprietaire> p) const override;
	private:
		std::shared_ptr<personne> _p1;
		std::shared_ptr<personne> _p2;
	friend proprietaire;
};

