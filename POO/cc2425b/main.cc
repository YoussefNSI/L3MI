#include "breakout.hh"
#include <iostream>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

int main() {
	coord largeur(200), hauteur(300);
	jeu j(hauteur);

	j.ajouter(std::make_unique<mur>(position_t(0, 0), taille_t(largeur, 10)));
	j.ajouter(std::make_unique<mur>(position_t(0, 10), taille_t(10, hauteur - 10)));
	j.ajouter(std::make_unique<mur>(position_t(largeur - 10, 10), taille_t(10, hauteur - 10)));

	for (int y = 0; y < 3; ++y) {
		int x(0);
		try {
			while (true) {
				j.ajouter(std::make_unique<bloc>(position_t(20 + x * 22, 40 + y * 22), ((x + y) % 5 == 4) ? bloc::type::indestructible : bloc::type::destructible));
				++x;
			}
		} catch (breakout_exception const&) {
		}
	}
	j.ajouter(std::make_unique<raquette>(position_t(largeur / 2 - 20, hauteur - 30), 40));
	j.ajouter(std::make_unique<balle>(position_t(largeur / 2, hauteur - 50), vitesse_t(0.7, -1.0)));
	sf::RenderWindow window(sf::VideoMode(largeur, hauteur), "Breakout");
	window.setKeyRepeatEnabled(false);
	window.setVerticalSyncEnabled(true);
	window.setFramerateLimit(90);

	/*
	for (auto const& i : j.objets()) {
		std::cout << (*i) << "\n";
	}
	*/
	sf::sleep(sf::seconds(20));
	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
			else if (event.type == sf::Event::KeyPressed) {
				if (event.key.code == sf::Keyboard::Right)
					j.bouger_raquette(raquette::direction::droite);
				else if (event.key.code == sf::Keyboard::Left) {
					j.bouger_raquette(raquette::direction::gauche);
				}
			}
			else if (event.type == sf::Event::KeyReleased)
				j.bouger_raquette(raquette::direction::stop);
		}
		auto nouvetat = j.evoluer();
		switch (nouvetat) {
			case jeu::etat::perdu:
				std::cout << "Vous avez perdu\n";
				window.close();
				break;
			case jeu::etat::gagne:
				std::cout << "Vous avez gagné\n";
				window.close();
				break;
			case jeu::etat::en_cours:
				window.clear(sf::Color::Black);
				for (auto const& i : j.objets()) {
					if (dynamic_cast<balle const*>(i.get())) {
						sf::CircleShape cir(i->taille().h() / 2);
						cir.setFillColor(sf::Color::White);
						cir.setPosition(i->position().x(), i->position().y());
						window.draw(cir);
					}
					else {
						sf::Color couleur;
						if (dynamic_cast<mur const*>(i.get()))
							couleur = sf::Color::Green;
						else if (dynamic_cast<bloc const*>(i.get())) {
							if (dynamic_cast<bloc const*>(i.get())->type_bloc() == bloc::type::destructible)
								couleur = sf::Color::Cyan;
							else
								couleur = sf::Color::Blue;
						}
						else
							couleur = sf::Color::Magenta;
						sf::RectangleShape rec(sf::Vector2f(i->taille().w(), i->taille().h()));
						rec.setFillColor(couleur);
						rec.setPosition(i->position().x(), i->position().y());
						window.draw(rec);
					}
				}
				window.display();
				break;
		}
	}
	return 0;
}
