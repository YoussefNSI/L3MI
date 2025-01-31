<?php
require 'selectionner-questions.php';
require 'selectionner-alternatives.php';

/*
 * sélectionne des enregistrements ("questions") de la table QUESTION et les enregistrements ("réponses")
 * de la table ALTERNATIVE qui leurs sont associés, puis renvoie le tout sous forme de tableau associatif.
 * Chaque élément du tableau correspond à une question et à ses réponses : sa clé est l'identifiant de la
 * question (colonne ID), sa valeur est un tableau indexé dont le premier élément est l'énoncé de la 
 * question (colonne ENONCE) et les éléments suivants sont les réponses représentées chacune sous forme 
 * de tableau associant les noms de colonnes REPONSE et SOLUTION aux valeurs de la réponse pour ces 
 * colonnes.
 * 
 * Le mode de sélection dépend de la valeur du paramètre $a :
 * - si $a est un entier, $a questions sont tirées aléatoirement dans la table,
 * - si $a est un tableau d'entier, les valeurs de $a correspondent aux identifiants des questions à
 * sélectionner,
 * - si $a ne remplit pas l'une de ces deux conditions ou si sa valeur est incorrecte, un tableau vide 
 * est renvoyé.
 * 
 * $a prend 5 comme valeur par défaut.
 * 
 * Exemple de tableau renvoyé avec $a=[3,5] :
 *
 * Array
 * (
 *  [3] => Array
 *      (
 *          [0] => <code>$x = null; echo gettype($x);</code>
 *          [1] => Array
 *              (
 *                  [REPONSE] => <code>NULL</code>
 *                  [SOLUTION] => 1
 *              )
 *          [2] => Array
 *              (
 *                  [REPONSE] => <code>boolean</code>
 *                  [SOLUTION] => 0
 *              )
 *          [3] => Array
 *              (
 *                  [REPONSE] => <code>undefined</code>
 *                  [SOLUTION] => 0
 *              )
 *      )
 *
 *  [5] => Array
 *      (
 *          [0] => <code>$x =''; var_dump(empty($x));</code>
 *          [1] => Array
 *              (
 *                  [REPONSE] => <code>bool(false)</code>
 *                  [SOLUTION] => 0
 *              )
 *          [2] => Array
 *              (
 *                  [REPONSE] => <code>bool(true)</code>
 *                  [SOLUTION] => 1
 *              )
 *      )
 *  )
 *
 * @param PDO $pdo L'objet de connexion à la base de données.
 *
 * @param integer|integer[] $a nombre de questions à tirer aléatoirement ou tableau des identifiants de
 * questions à sélectionner.
 *  
 * @return mixed[][] Un tableau de questions-réponses.
 */
function selectionnerQuestionsAlternatives($pdo, $a=5)
{
    // TODO
}
?>
