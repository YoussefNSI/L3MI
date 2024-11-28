#include "snake.hh"
#include <iostream>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>


int main() {
	jeu j(80, 60);

	sf::RenderWindow window(sf::VideoMode(j.largeur()*4, j.hauteur()*4), "Snake");
	window.setKeyRepeatEnabled(false);
	window.setVerticalSyncEnabled(true);
	window.setFramerateLimit(2);

	for (unsigned int i(0); i < 20; ++i)
		j.obstacle_ajoute(j.position_aleatoire());
	j.bonus_ajoute(15);

	affichage aff(j.largeur(), j.hauteur());
	j.affichage_remplir(aff);

	unsigned int decompte(0);
	etatjeu etat(etatjeu::encours);
	while (window.isOpen() && (etat == etatjeu::encours)) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
			else if (event.type == sf::Event::KeyPressed) {
				if (event.key.code == sf::Keyboard::Left)
					j.serpent_direction(1, direction::gauche);
				else if (event.key.code == sf::Keyboard::Right)
					j.serpent_direction(1, direction::droite);
				if (event.key.code == sf::Keyboard::Up)
					j.serpent_direction(1, direction::haut);
				else if (event.key.code == sf::Keyboard::Down)
					j.serpent_direction(1, direction::bas);
				if (event.key.code == sf::Keyboard::Q)
					j.serpent_direction(0, direction::gauche);
				else if (event.key.code == sf::Keyboard::D)
					j.serpent_direction(0, direction::droite);
				if (event.key.code == sf::Keyboard::Z)
					j.serpent_direction(0, direction::haut);
				else if (event.key.code == sf::Keyboard::S)
					j.serpent_direction(0, direction::bas);
			}
		}
		if (decompte++ == 20)
			decompte = 0;
		etat = j.tour_de_jeu(decompte == 10);
		j.affichage_remplir(aff);

		window.clear(sf::Color::Black);
		for (coord x(0); x < j.largeur(); ++x) {
			for (coord y(0); y < j.hauteur(); ++y) {
				typecase tc(aff.acces(position(x, y)));
				if ((x == 0) || (y == 0) || (x == (j.largeur() - 1) || (y == (j.hauteur() - 1))))
					tc = typecase::obstacle;
				sf::RectangleShape rec(sf::Vector2f(4.0, 4.0));
				sf::Color col;
				switch (tc) {
					case typecase::serpent1:
						col = sf::Color::Blue;
						break;
					case typecase::serpent2:
						col = sf::Color::Yellow;
						break;
					case typecase::obstacle:
						col = sf::Color::Red;
						break;
					case typecase::bonus:
						col = sf::Color::Green;
						break;
					case typecase::vide:
						col = sf::Color::Black;
						break;
				}
				rec.setFillColor(col);
				rec.setPosition(x * 4, y * 4);
				window.draw(rec);
			}
		}
		window.display();
	}
	aff.afficher(std::cout);
	switch (etat) {
		case etatjeu::encours:
			std::cout << "En cours\n";
			break;
		case etatjeu::vainqueur1:
			std::cout << "Joueur 1 gagne\n";
			break;
		case etatjeu::vainqueur2:
			std::cout << "Joueur 2 gagne\n";
			break;
		case etatjeu::aucunvainqueur:
			std::cout << "Egalité\n";
			break;
	}

	return 0;
}
