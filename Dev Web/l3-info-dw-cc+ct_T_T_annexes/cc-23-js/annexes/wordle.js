//////////////////////////////////////////////////////
// DONNEES ET VARIABLES D'ETAT


//////////////////////////////////////////////////////
// FONCTIONS

// initialise ou réinitialise les variables d'état, le texte et la couleur de fond des cellules,
// celle du pied de page HTML (footer) et son contenu (chaîne vide)
function reset() {}

// colorie le fond de la cellule (i,j) selon la correction qui lui est attribuée
function colorie(i, j) {}

// corrige la ligne courante en mettant à jour les variables d'état, sa mise en forme
// et, en fin de partie, le pied de page en y incoporant message puis statistiques
function corrigerLigne() {}

// Requête en HTTP POST le script wordle_stats.php en communiquant la grille de lettres
// au format JSON et le mot-solution.
// Affiche les résultats sous forme d'items de liste dans le pied de page.
function statistiques() {}


//////////////////////////////////////////////////////
// ECOUTEURS

// écouteur des clics sur le bouton "Go !"

// écouteur du clavier