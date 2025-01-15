#pragma once

#include "jeu.hh"
#include <filesystem>
#include <array>
#include <SFML/Graphics.hpp>

class affichage {
	public:
	affichage(std::filesystem::path const& rep_sprites, jeu const& j);

	void dessiner_plateau();
	void dessiner_explosions();
	void dessiner_mobiles();
	void mettre_a_jour_affichage();

	bool fenetre_ouverte() const;

	bool lire_action(joueur_action& ja, joueur_numero& jn);

	private:
	std::filesystem::path _repertoire_sprites;

	std::array<sf::Texture, 17> _explosions_textures;
	std::array<sf::Sprite, 17> _explosions_sprites;

	std::array<sf::Texture, 6> _blocs_textures;
	std::array<sf::Sprite, 6> _blocs_sprites;

	const std::size_t bloc_obstacle = 0;
	const std::size_t bloc_fond = 1;
	const std::size_t bloc_bombe = 2;
	const std::size_t bloc_bonus_bombe = 3;
	const std::size_t bloc_bonus_portee = 4;
	const std::size_t bloc_bonus_vie = 5;

	struct joueur_ts {
		std::array<sf::Texture, 16> _textures;
		std::array<sf::Sprite, 16> _sprites;
	};
	std::array<joueur_ts, 2> _joueurs;

	jeu const& _jeu;
	sf::RenderWindow _window;
};
