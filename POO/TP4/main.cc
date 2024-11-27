#include "bomberman_v2.hh"
#include <filesystem>
#include <array>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

int main() {
	const unsigned int sprite_w(64), sprite_h(48);
	// Modifier cette ligne et mettre comme valeur le répertoire contenant le sprites (exemple "/home/moi/tpcpp/bomberman/sprites". Si on laisse à vide, cela désignera la répertoire courant. Attention, par répertoire courant, on entend ici le répertoire DE L'EXECUTABLE et non celui de vos sources : il vous faudra donc copier les fichiers PNG dans le répertoire "build-NomDuProjet" créé par QtCreator.
	std::filesystem::path path_sprites("/home/moi/tccpp/bomberman/sprites");
	// Chargement des éléments graphique d'explosions
	std::array<sf::Texture, 17> textures_expl;
	std::array<sf::Sprite, 17> sprites_expl;
	for (std::size_t i(0); i < 16; ++i) {
		auto file_sprite(path_sprites / (std::string("expl0") + static_cast<char>((i <= 9) ? '0' + i : 'a' + i - 10) + ".png"));
		if (!textures_expl[i].loadFromFile(file_sprite)) {
			std::cerr << "Sprite introuvable : " << file_sprite;
			return 1;
		}
		sprites_expl[i].setTexture(textures_expl[i]);
	}
	// Chargement des autres éléments graphiques (bonus, obstacle, fond).
	sf::Texture texture_obstacle, texture_fond, texture_bombe, texture_bonus_bombe, texture_bonus_portee, texture_bonus_vie;
	texture_obstacle.loadFromFile(path_sprites / "bricks.png");
	texture_fond.loadFromFile(path_sprites / "button_floor.png");
	texture_bombe.loadFromFile(path_sprites / "bomb_0.png");
	texture_bonus_bombe.loadFromFile(path_sprites / "bonus_bomb.png");
	texture_bonus_portee.loadFromFile(path_sprites / "bonus_range.png");
	texture_bonus_vie.loadFromFile(path_sprites / "bonus_extra.png");
	sf::Sprite sprite_obstacle(texture_obstacle), sprite_fond(texture_fond), sprite_bombe(texture_bombe), sprite_bonus_bombe(texture_bonus_bombe), sprite_bonus_portee(texture_bonus_portee), sprite_bonus_vie(texture_bonus_vie);

	// Initialisation du plateau de jeu.
	const int W(18), H(15);
	plateau p(position(W, H));
	for (coord x = 0; x < W; ++x) {
		p.ajouter(position(x, 0), /* Un obstacle */);
		p.ajouter(position(x, H - 1), /* Un obstacle */);
	}
	for (coord y = 0; y < H; ++y) {
		p.ajouter(position(0, y), /* Un obstacle */);
		p.ajouter(position(W - 1, y), /* Un obstacle */);
	}
	p.ajouter(position(3, 8), /* Un obstacle */);
	p.ajouter(position(2, 11), /* Un obstacle */);
	p.ajouter(position(3, 6), /* Une bombe de largeur 5 */);
	p.ajouter(position(11, 12), /* Un bonus de type vie_supplementaire */);
	p.ajouter(position(11, 11), /* Un bonus de type amelioration_nombre */);
	p.ajouter(position(11, 10), /* Un bonus de type amelioration_portee */);

	// Boucle de mise à jour de l'état du jeu.
	sf::RenderWindow window(sf::VideoMode(sprite_w * W, sprite_h * H), "Bomberman");
	window.setKeyRepeatEnabled(false);
	window.setVerticalSyncEnabled(true);
	window.setFramerateLimit(2); // L'affichage est fait ici à 3 images par seconde, ce qui est utile pour bien voir la décomposition de ce qui se passe. Vous pouvez modifier la valeur pour avoir quelque chose de plus réactif.

	std::size_t etat(0);
	while (window.isOpen()) {
		etat++;
		if (etat == 10) { // A l'état 10, on rajoute deux nouvelles bombes.
			p.ajouter(position(3, 11), /* Une bombe de largeur 8 */);
			p.ajouter(position(8, 6), /* Une bombe de largeur 6 */);
		}
		else if (etat == 15) // Et une autre à l'état 15.
			p.ajouter(position(8, 12), /* Une bombe de largeur 8 */);
		p.etat_suivant();
		p.sortie_graphique(std::cout); // Simplement pour aider au débogage, on affiche le plateau sur le flux standard en plus de l'affichage dans la fenêtre graphique ci-dessous.
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
		}
		window.clear(sf::Color(128,128,128));
		for (coord y = 0; y < H; ++y) {
			for (coord x = 0; x < W; ++x) {
				auto const& ent(p.acces(position(x, y))); // On accède à l'entité se trouvant en x,y ...
				if (ent) { // ... et s'il y en a une ...
					sf::Sprite* spr(nullptr);
					switch (ent->symbole()) { // ...on détermine dans spr le sprite qui doit être affiché, en fonction du symbole.
						case '#': {
							spr = &sprite_obstacle;
							break;
						}
						case 'B': {
							sprite_fond.setPosition(x * sprite_w, y * sprite_h); // Dans le cas où c'est une bombe, on dessine le fond et la suite du code dessinera la bombe.
							window.draw(sprite_fond);
							spr = &sprite_bombe;
							break;
						}
						case 'P': {
							spr = &sprite_bonus_portee;
							break;
						}
						case 'N': {
							spr = &sprite_bonus_bombe;
							break;
						}
						case 'V': {
							spr = &sprite_bonus_vie;
							break;
						}
						case 'E': { // Dans le cas où c'est une bombe en cours d'explosion (symbole E), on affiche le fond (plutôt que la bombe), et l'explosion sera affichée par dessus par la code plus bas.
							spr = &sprite_fond;
							break;
						}
					}
					if (spr) {
						spr->setPosition(x * sprite_w, y * sprite_h);
						window.draw(*spr);
					}
				}
				else {
					sprite_fond.setPosition(x * sprite_w, y * sprite_h);
					window.draw(sprite_fond); // S'il n'y a rien, on dessine le fond.
				}
				// affichage des explosions
				auto ex(p.acces_explosions().acces(position(x, y)));
				if (ex) {
					std::size_t indicesprite;
					indicesprite = 0; // Ligne à modifier pour afficher le bon élément graphique, ici, ça affiche systématiquement (les deux lignes ci-dessous) une petite explosion.
					sprites_expl.at(indicesprite).setPosition(x * sprite_w, y * sprite_h);
					window.draw(sprites_expl.at(indicesprite));
				}
			}
	}
		window.display();
	}
	return 0;
}
