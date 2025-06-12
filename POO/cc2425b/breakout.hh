#pragma once

#include "taillepos.hh"
#include <list>
#include <memory>
#include <exception>
#include <ostream>

class objet {
	public:
	objet(position_t const& p)
		: _pos(p) { }
	virtual ~objet() = default;

	virtual std::unique_ptr<objet> clone() const = 0;

	position_t const& position() const { return _pos; }
	virtual taille_t taille() const = 0;

	void modifier_position(position_t const& p) { _pos = p; }
	virtual bool detruit_quand_collision() const;

	bool collision(position_t const& p1, objet const& o2);

	virtual void sortie_flux(std::ostream & os) const;

	protected:
	virtual std::string type_objet() const = 0;

	private:
	position_t _pos;
};
std::ostream& operator<<(std::ostream& os, objet const& o);

class mur : public objet {
	public:
	mur(position_t const& p, taille_t const& t);
	std::unique_ptr<objet> clone() const override;
	taille_t taille() const override;
	protected:
	std::string type_objet() const override;
	private:
	taille_t _taille;
};

class bloc : public objet {
	public:
	enum type {
		destructible,
		indestructible
	};
	bloc(position_t const& p, type t)
		: objet(p)
		, _typebloc(t) {
	}
	std::unique_ptr<objet> clone() const override;

	taille_t taille() const override;

	bool detruit_quand_collision() const override;

	type type_bloc() const { return _typebloc; }

	protected:
	std::string type_objet() const override;
	private:
	type _typebloc;
};

class objet_en_mouvement : public objet {
	public:
	objet_en_mouvement(position_t const& p)
		: objet(p) { }

	virtual vitesse_t vitesse() const = 0;

	position_t calculer_deplacement() const;
	virtual void appliquer_deplacement_simple();
	virtual void appliquer_deplacement_collision(objet const & o) =0;
	void sortie_flux(std::ostream & os) const override;
};

class balle : public objet_en_mouvement {
	public:
	balle(position_t const& p, vitesse_t const & vi)
		: objet_en_mouvement(p)
		, _vitesse(vi) { }
	std::unique_ptr<objet> clone() const override;

	taille_t taille() const override { return taille_t(5, 5); }
	vitesse_t vitesse() const override { return _vitesse; }
	void appliquer_deplacement_collision(objet const & o) override;

	protected:
	std::string type_objet() const override;
	private:
	vitesse_t _vitesse;
};

class raquette : public objet_en_mouvement {
	public:
	enum class direction {
		droite,
		gauche,
		stop
	};
	raquette(position_t const& p, coord largeur)
		: objet_en_mouvement(p)
		, _largeur(largeur)
		, _dir(direction::stop) { }
	std::unique_ptr<objet> clone() const override;

	taille_t taille() const override { return taille_t(_largeur, 5); }
	vitesse_t vitesse() const override;
	void appliquer_deplacement_collision(objet const & o) override;

	void modifier_direction(direction d);

	protected:
	std::string type_objet() const override;
	private:
	coord _largeur;
	direction _dir;
};

class breakout_exception : public std::exception {
	public:
	enum class type {
		plusieurs_raquettes,
		trop_petit,
		impossible_ici,
		pas_de_balle
	};
	breakout_exception(type t)
		: std::exception()
		, _tp(t) { }

	char const* what() const noexcept override;

	private :
	type _tp;
};

class jeu {
	public:
	enum class etat {
		en_cours,
		perdu,
		gagne
	};
	jeu(coord hauteur);
	jeu(jeu const& j);
	jeu& operator=(jeu const& j);

	void ajouter(std::unique_ptr<objet> obj);

	void bouger_raquette(raquette::direction d);
	balle const & acces_balle() const;
	std::size_t nombre_blocs_destructibles() const;
	etat evoluer();

	std::list<std::unique_ptr<objet>> const & objets() const; // Non demandé, utile pour l'affichage SFML
	private:
	std::list<std::list<std::unique_ptr<objet>>::iterator> deplacer();
	void ensemble_objets_ajouter(std::list<std::list<std::unique_ptr<objet>>::iterator> &c, std::list<std::unique_ptr<objet>>::iterator i);
	void effacer_objets(std::list<std::list<std::unique_ptr<objet>>::iterator> const & lc);

	private:
	std::list<std::unique_ptr<objet>> _objets;
	coord _hauteur;
};
