#pragma once
#include <memory>
#include "proprietaire.hh"

class compte {
	public:
		compte(std::shared_ptr<proprietaire> prop, float m):_prop(prop), _montant(m), _numero(++_compteur) {}
		virtual ~compte() =default;
		bool verser(float m);
		bool retirer(float m);
		bool virer(compte & dest, float m);
		void appliquerinterets();
		float montant() const { return _montant; }
		unsigned int numero() const { return _numero; }
		std::shared_ptr<proprietaire> prop() const { return _prop; }
	protected:
		virtual bool montantautorise(float m) const =0;
		virtual float taux() const;
	private:
		std::shared_ptr<proprietaire> _prop;
		float _montant;
		unsigned int _numero;
		static unsigned int _compteur;
};

class comptecourant: public compte {
	public:
		// Si on voulait vérifier que le montant est autorisé, il faudrait, comme pour les propriétaires, mettre en place une méthode de fabrique (plus constructeur privé) vérifiant le montant et créant le compte uniquement dans le cas où le montant est correct. 
		comptecourant(std::shared_ptr<proprietaire> prop, float m):compte(prop, m), _decouvertautorise(0) {}
		void modifierdecouvert(float nd) { _decouvertautorise = nd; }
	protected:
		bool montantautorise(float m) const override { return m >= _decouvertautorise; } 
	private:
		float _decouvertautorise;
};

class ldd: public compte {
	public:
		// Idem
		ldd(std::shared_ptr<proprietaire> prop, float m): compte(prop, m) {}
		float taux() const override { return _taux; }
		static void modifiertaux(float taux) { _taux = taux; }
	protected:
		bool montantautorise(float m) const override { return (m >= 15) && (m <= 12000); } 
	private:
		static float _taux;
};

class lep: public compte {
	public:
		// Idem + il faudrait vérifier que le propriétaire est une personne, et cela aussi peut être vérifié dans la fabrique.
		lep(std::shared_ptr<proprietaire> prop, float m): compte(prop, m) {}
		float taux() const override { return _taux; }
		static void modifiertaux(float taux) { _taux = taux; }
	protected:
		bool montantautorise(float m) const override { return (m >= 30) && (m <= 7700); } 
	private:
		static float _taux;
};
