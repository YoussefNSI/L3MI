<?php
// Création du tableau [..., "D" => ["Prix" => 22.71,"Taux" => 0.05], ...]
$articles = range("A", "K");
$taux = [
    0.05,
    0.10,
    0.20
];
$articles = array_fill_keys($articles, 0);

/**
 * renvoie un tableau au format ["Prix" => 22.71,"Taux" => 0.05]
 * où le prix est tiré aléatoirement dans [1,100]
 * et le taux est tiré aléatoirement dans le tableau de taux $tab_taux
 * 
 * NB. $tab_taux est passé en paramètre plutôt qu'utilisé comme une variable globale.
 * Ce qui permet d'utiliser indirectement cette fonction nommée comme fonction de rappel
 * dans les fonctions de type array_*.
 * 
 * return array
 **/

function creer_prix_articles($tab_taux)
{
    return array(
        "Prix" => ((float) rand(1, 10000) / 100),
        "Taux" => $tab_taux[rand(0, 2)]
    );
}

$prix_taux = array_map(function($a) use($taux) { return creer_prix_articles($taux); }, $articles);
?>