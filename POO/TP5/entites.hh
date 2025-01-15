#pragma once
#include "position.hh"

using identifiant = unsigned int;
class entite {
	public:
	entite();
	virtual ~entite() = default;
	virtual char symbole() const = 0;
	identifiant id() const { return _id; }
	virtual void sortie_flux(std::ostream& os) const;
	virtual bool accepte_joueur() const =0;

	private:
	static identifiant _compteur;
	identifiant _id;
};
std::ostream& operator<<(std::ostream& os, entite const& e);

enum class bombe_etat {
	decompte,
	explosion,
};
class bombe : public entite {
	public:
	bombe(coord largeur);
	char symbole() const override;
	void sortie_flux(std::ostream& os) const override;
	bool accepte_joueur() const override;

	bombe_etat etat() const { return _etat; }
	coord largeur() const { return _largeur; }
	coord rebours() const { return _decompte; }
	bool etat_suivant();
	void exploser();
	static void modifier_decompte(coord nd);
	private:
	static coord _decompte_defaut;
	coord _decompte;
	coord _largeur;
	bombe_etat _etat;
};

class obstacle : public entite {
	public:
	obstacle()
		: entite() { }
	char symbole() const override;
	bool accepte_joueur() const override;
};

enum class bonus_type {
	amelioration_portee,
	amelioration_nombre,
	vie_supplementaire
};
class bonus : public entite {
	public:
	bonus(bonus_type bt)
		: entite(), _type(bt) { }
	char symbole() const override;
	void sortie_flux(std::ostream& os) const override;
	bool accepte_joueur() const override;

	bonus_type type() const { return _type; }
	private:
	bonus_type _type;
};

