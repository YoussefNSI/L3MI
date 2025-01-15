#pragma once

#include "mobiles.hh"
#include "plateau.hh"
#include <memory>

enum class joueur_action {
	haut,
	bas,
	droite,
	gauche,
	stop,
	action
};
using joueur_numero = std::uint8_t;

class jeu {
	public:
	jeu();

	class plateau const& plateau() const { return _plateau; }
	std::vector<std::unique_ptr<mobile>> const & mobiles() const { return _mobiles; }
	void plateau_initialiser();
	void bombe_ajouter(position const& p, coord largeur);
	void joueur_ajouter(position const& p, joueur_numero jn);
	void ennemi_ajouter(position const& p);
	void bombe_ajouter(coord largeur);
	void joueur_ajouter(joueur_numero jn);
	void ennemi_ajouter();
	void etat_suivant();

	void ajouter_action(joueur_action ac, joueur_numero jn);

	private:
	bool contient_mobile(position const& p) const;
	position position_aleatoire() const;
	void ennemis_bouger();

	private:
	class plateau _plateau;
	std::vector<std::unique_ptr<mobile>> _mobiles;
	unsigned int _horloge;
};
