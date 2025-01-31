<?php

/*
 * sélectionne des enregistrements ("questions") de la table QUESTION et renvoie ces questions sous 
 * forme de tableau associatif. Chaque élément du tableau correspond à une question et a pour clé 
 * l'identifiant de la question (colonne ID) et pour valeur l'énoncé de la question (colonne ENONCE).
 *
 * Le mode de sélection dépend de la valeur du paramètre $a :
 * - si $a est un entier, $a questions sont tirées aléatoirement dans la table,
 * - si $a est un tableau d'entier, les valeurs de $a correspondent aux identifiants des questions à
 * sélectionner.
 *
 * $a prend 5 comme valeur par défaut.
 *
 * Toute valeur incorrecte pour $a (i.e., type incorrect, valeur entière négative, valeur entière
 * supérieure au nombre d'enregistrements de QUESTION, tableau dont une valeur ne correspond à aucun
 * identifiant) lève une exception.
 *
 * Exemple de tableau renvoyé avec $a=[1,11,26] :
 *
 * Array
 * (
 *  [1] => Que fait l'instruction <code>include('file.php');</code> si <code>file.php</code> n'existe pas ?
 *  [11] => <code>echo 234 <=> 123;</code>
 *  [26] => <code>$x='\"1'; echo "$x";</code>
 * )
 *
 * @param PDO $pdo L'objet de connexion à la base de données.
 *
 * @param integer|integer[] $a nombre de questions à tirer aléatoirement ou tableau des identifiants de
 * questions à sélectionner.
 *
 * @throws InvalidArgumentException si $a ne remplit pas les conditions énoncées ci-dessus.
 *
 * @return string[] Un tableau dont chaque élément associe l'énoncé d'une question à son identifiant.
 */
function selectionnerQuestions($pdo, $a = 5)
{
    // TODO
}
?>