<?php
// Création du tableau [..., "D" => ["Prix" => 22.71,"Taux" => 0.05], ...]



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


$tab_taux = [0.05, 0.10, 0.20];
function creer_prix_articles($tab_taux)
{
    $tab_article = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"];
    $tab_articles = array_map(function() use ($tab_taux) {
        return ["Prix" => rand(1, 100), "Taux" => $tab_taux[array_rand($tab_taux)]];
    }, array_flip($tab_article));
    return $tab_articles;
}

// initialisation de $prix_taux
$prix_taux = creer_prix_articles($tab_taux);
?>