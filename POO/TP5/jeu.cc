#include "jeu.hh"
#include <cstdlib>

jeu::jeu()
	: _plateau(position(18, 15)), _horloge(0) {
}

void jeu::plateau_initialiser() {
	for (coord x(0); x < _plateau.taille().x(); ++x) {
		_plateau.ajouter(position(x, 0), std::make_unique<obstacle>(obstacle()));
		_plateau.ajouter(position(x, _plateau.taille().y() - 1), std::make_unique<obstacle>());
	}
	for (coord y(0); y < _plateau.taille().y(); ++y) {
		_plateau.ajouter(position(0, y), std::make_unique<obstacle>());
		_plateau.ajouter(position(_plateau.taille().x() - 1, y), std::make_unique<obstacle>());
	}
	for (unsigned int i(0); i < 40; ++i)
		_plateau.ajouter(position_aleatoire(), std::make_unique<obstacle>());
	for (unsigned int i(0); i < 5; ++i)
		_plateau.ajouter(position_aleatoire(), std::make_unique<bonus>(bonus_type::amelioration_nombre));
	for (unsigned int i(0); i < 5; ++i)
			_plateau.ajouter(position_aleatoire(), std::make_unique<bonus>(bonus_type::amelioration_portee));
	for (unsigned int i(0); i < 2; ++i)
		_plateau.ajouter(position_aleatoire(), std::make_unique<bonus>(bonus_type::vie_supplementaire));
}

void jeu::bombe_ajouter(const position& p, coord largeur) {
	_plateau.ajouter(p, std::make_unique<bombe>(largeur));
}

void jeu::joueur_ajouter(const position& p, joueur_numero jn) {
	_mobiles.push_back(std::make_unique<joueur>(p, jn));
}

void jeu::ennemi_ajouter(const position& p) {
	_mobiles.push_back(std::make_unique<ennemi>(p));
}

void jeu::bombe_ajouter(coord largeur) {
	bombe_ajouter(position_aleatoire(), largeur);
}

void jeu::joueur_ajouter(joueur_numero jn) {
	joueur_ajouter(position_aleatoire(), jn);
}

void jeu::ennemi_ajouter() {
	ennemi_ajouter(position_aleatoire());
}

void jeu::etat_suivant() {
	++_horloge;
	if ((_horloge % 8) == 0)
		_plateau.etat_suivant();
	for (auto& i : _mobiles)
		i->etat_suivant(_plateau);
	if ((_horloge % 50) == 0)
		ennemis_bouger();
}

void jeu::ajouter_action(joueur_action ac, joueur_numero jn) {
	for (auto& i : _mobiles) {
		auto ijoueur(dynamic_cast<joueur*>(i.get()));
		if (ijoueur && ijoueur->numero() == jn) {
			switch (ac) {
				case joueur_action::haut:
					ijoueur->fixer_dir_voulue(direction::haut);
					break;
				case joueur_action::bas:
					ijoueur->fixer_dir_voulue(direction::bas);
					break;
				case joueur_action::droite:
					ijoueur->fixer_dir_voulue(direction::droite);
					break;
				case joueur_action::gauche:
					ijoueur->fixer_dir_voulue(direction::gauche);
					break;
				case joueur_action::stop:
					ijoueur->fixer_dir_voulue(direction::stop);
					break;
				case joueur_action::action:
					if (_plateau.libre(ijoueur->pos()))
						bombe_ajouter(ijoueur->pos(), ijoueur->portee_bombes());
					break;
			}
			break;
		}
	}
}

bool jeu::contient_mobile(const position& p) const {
	for (auto const& i : _mobiles)
		if (i->pos() == p)
			return true;
	return false;
}

position jeu::position_aleatoire() const {
	while (true) {
		position pos(rand() % _plateau.taille().x() + 1, rand() % _plateau.taille().y() + 1);
		if (_plateau.libre(pos) && !contient_mobile(pos))
			return pos;
	}
}

void jeu::ennemis_bouger() {
	for (auto& i : _mobiles) {
		auto iennemi(dynamic_cast<ennemi*>(i.get()));
		if (iennemi)
			iennemi->fixer_dir_voulue(ennemi::direction_aleatoire());
	}
}
