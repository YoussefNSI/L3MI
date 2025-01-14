#include "jeu.hh"
#include "affichage.hh"

int main() {

	jeu j;
	j.plateau_initialiser();
	j.joueur_ajouter(0);
	for (int i(0); i < 8; ++i)
		j.ennemi_ajouter();

	affichage aff("/home/genest/Enseignements/l3_poo/exercices/tp5/3_bomberman_v3/sprites", j);

	while (aff.fenetre_ouverte()) {
		joueur_action ja;
		joueur_numero jn;
		while (aff.lire_action(ja, jn)) {
			j.ajouter_action(ja, jn);
		}

		j.etat_suivant();

		aff.dessiner_plateau();
		aff.dessiner_explosions();
		aff.dessiner_mobiles();
		aff.mettre_a_jour_affichage();
	}
	return 0;
}
