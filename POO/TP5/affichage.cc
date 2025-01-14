#include "affichage.hh"
#include <iostream>
#include <cmath>
#include <algorithm>

affichage::affichage(const std::filesystem::path& rep_sprites, jeu const & je)
	: _repertoire_sprites(rep_sprites), _jeu(je), _window(sf::VideoMode(je.plateau().taille().x()*plateau::bloc_w, je.plateau().taille().y()*plateau::bloc_h), "Bomberman") {
	// Chargement des explosions
	for (std::size_t i(0); i < 16; ++i) {
		auto file_sprite(_repertoire_sprites / (std::string("expl0") + static_cast<char>((i <= 9) ? '0' + i : 'a' + i - 10) + ".png"));
		if (!_explosions_textures[i].loadFromFile(file_sprite)) {
			std::cerr << "Sprite introuvable : " << file_sprite << "\n";
			exit(1);
		}
		_explosions_sprites[i].setTexture(_explosions_textures[i]);
	}
	{	// Chargement des blocs
		std::array<const char*, 6> blocs_noms { "bricks", "button_floor", "bomb_0", "bonus_bomb", "bonus_range", "bonus_extra" };
		for (std::size_t i(0); i < blocs_noms.size(); ++i) {
			auto file_sprite(_repertoire_sprites / (std::string(blocs_noms[i]) + ".png"));
			if (!_blocs_textures[i].loadFromFile(file_sprite)) {
				std::cerr << "Sprite introuvable : " << file_sprite << "\n";
				exit(1);
			}
			_blocs_sprites[i].setTexture(_blocs_textures[i]);
		}
	}
	{	// Chargement des joueurs
		std::array<const char*, 2> joueurs_noms { "normal", "tall" };
		std::array<const char*, 16> suffixes {"D_0", "D_1", "D_2", "D_3", "U_0", "U_1", "U_2", "U_3", "L_0", "L_1", "L_2", "L_3", "R_0", "R_1", "R_2", "R_3"};
		for (std::size_t i(0); i < joueurs_noms.size(); ++i) {
			for (std::size_t j(0); j < suffixes.size(); ++j) {
				auto file_sprite(_repertoire_sprites / (std::string(joueurs_noms[i]) + "_" + suffixes[j] + ".png"));
				if (!_joueurs[i]._textures[j].loadFromFile(file_sprite)) {
					std::cerr << "Sprite introuvable : " << file_sprite << "\n";
					exit(1);
				}
				_joueurs[i]._sprites[j].setTexture(_joueurs[i]._textures[j]);
			}
		}
	}
	_window.setKeyRepeatEnabled(false);
	_window.setVerticalSyncEnabled(true);
	_window.setFramerateLimit(60);
}

void affichage::dessiner_plateau() {
	_window.clear(sf::Color(128, 128, 128));
	auto const& taille(_jeu.plateau().taille());
	for (coord y = 0; y < taille.y(); ++y) {
		for (coord x = 0; x < taille.x(); ++x) {
			std::size_t indicesprite(bloc_fond); // Ce qu'on affichera si on ne trouve rien d'autre.
			auto const& ent(_jeu.plateau().acces(position(x, y))); // On accède à l'entité se trouvant en x,y ...
			if (ent) { // ... et s'il y en a une ...
				std::initializer_list<std::pair<char, std::size_t>> symbol_to_sprite {
					{ '#', bloc_obstacle }, { 'B', bloc_bombe }, { 'P', bloc_bonus_portee }, { 'N', bloc_bonus_bombe }, { 'V', bloc_bonus_vie }, { 'E', bloc_fond }
				};
				auto entsymbole(ent->symbole());
				for (auto const& i : symbol_to_sprite)
					if (i.first == entsymbole) {
						indicesprite = i.second;
						break;
					}
				if (entsymbole == 'B') {
					_blocs_sprites[bloc_fond].setPosition(x * plateau::bloc_w, y * plateau::bloc_h); // Dans le cas où c'est une bombe, on dessine aussi le fond.
					_window.draw(_blocs_sprites[bloc_fond]);
				}
			}
			_blocs_sprites[indicesprite].setPosition(x * plateau::bloc_w, y * plateau::bloc_h);
			_window.draw(_blocs_sprites[indicesprite]);
		}
	}
}

void affichage::dessiner_explosions() {
	auto const& taille(_jeu.plateau().taille());
	for (coord y = 0; y < taille.y(); ++y) {
		for (coord x = 0; x < taille.x(); ++x) {
			auto ex(_jeu.plateau().acces_explosions().acces(position(x, y)));
			if (ex) {
				std::size_t indicesprite;
				if (ex == et_simple)
					indicesprite = 0;
				else
					indicesprite = ex & (et_simple - 1);
				_explosions_sprites.at(indicesprite).setPosition(x * plateau::bloc_w, y * plateau::bloc_h);
				_window.draw(_explosions_sprites.at(indicesprite));
			}
		}
	}
}

std::size_t sprite_joueur_indice(direction d, signed short dx, signed short dy) {
	switch (d) {
		case direction::haut:
			return 4 + ((dy + plateau::bloc_h) / 16) % 4;
		case direction::stop:
		case direction::bas:
			return 0 + ((dy + plateau::bloc_h) /16) % 4;
		case direction::droite:
			return 12 + ((dx + plateau::bloc_w) / 16) % 4;
		case direction::gauche:
			return 8 + ((dx + plateau::bloc_w) / 16) % 4;
			break;
	}
	return 0;
}

void affichage::dessiner_mobiles() {
	for (auto const& i : _jeu.mobiles()) {
		std::size_t indicesprite(_joueurs.size() - 1); // Par défaut, le dernier, qui est le sprite de l'ennemi.
		auto ijoueur(dynamic_cast<class joueur const*>(i.get()));
		if (ijoueur)
			indicesprite = std::min(static_cast<std::size_t>(ijoueur->numero()), _joueurs.size() - 2);
		auto & spr(_joueurs.at(indicesprite)._sprites.at(sprite_joueur_indice(i->dir_actuelle(), i->dx(), i->dy())));
		spr.setPosition(i->x() + plateau::bloc_w/2 - spr.getGlobalBounds().width / 2, i->y() + plateau::bloc_h/2 - spr.getGlobalBounds().height / 2 -15);
		_window.draw(spr);
	}
}

void affichage::mettre_a_jour_affichage() {
	_window.display();
}

bool affichage::fenetre_ouverte() const {
	return _window.isOpen();
}

bool affichage::lire_action(joueur_action& ja, joueur_numero& jn) {
	sf::Event e;
	if (_window.pollEvent(e)) {
		if (e.type == sf::Event::Closed) {
			_window.close();
		}
		else if (e.type == sf::Event::KeyPressed) {
			std::initializer_list<std::tuple<sf::Keyboard::Key, joueur_action, joueur_numero>> keys {
				{ sf::Keyboard::Left, joueur_action::gauche, 0 }, { sf::Keyboard::Right, joueur_action::droite, 0 }, { sf::Keyboard::Up, joueur_action::haut, 0 }, { sf::Keyboard::Down, joueur_action::bas, 0 }, { sf::Keyboard::Home, joueur_action::stop, 0 }, { sf::Keyboard::Space, joueur_action::action, 0 }
			};
			for (auto const& i : keys)
				if (e.key.code == std::get<0>(i)) {
					ja = std::get<1>(i);
					jn = std::get<2>(i);
					return true;
				}
		}
		return lire_action(ja, jn);
	}
	else
		return false;
}
