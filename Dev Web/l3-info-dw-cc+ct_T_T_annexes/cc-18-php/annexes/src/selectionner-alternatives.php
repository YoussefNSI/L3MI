<?php

/*
 * renvoie sous forme de tableau indexé les enregistrements de la table ALTERNATIVE qui 
 * correspondent à la question d'identifiant $id_question (colonne ID_QUESTION). Chaque élément 
 * du tableau est lui-même un tableau associant les clés "REPONSE" et "SOLUTION" aux valeurs de 
 * l'enregistrement pour les colonnes REPONSE et SOLUTION.
 * 
 * Exemple de tableau renvoyé pour une question :
 *
 * Array
 * (
 *  [0] => Array
 *      (
 *          [REPONSE] => <code>-1</code>
 *          [SOLUTION] => 0
 *      )
 *
 *  [1] => Array
 *      (
 *          [REPONSE] => <code>0</code>
 *          [SOLUTION] => 0
 *      )
 *
 *  [2] => Array
 *      (
 *          [REPONSE] => <code>1</code>
 *          [SOLUTION] => 1
 *      )
 * )
 *
 * @param int $id_question L'identifiant de la question.
 *
 * @param PDO $pdo L'objet de connexion à la base de données.
 *
 * @return mixed[][] Le tableau indexé des réponses stockées pour la question.
 */
function selectionnerAlternatives($id_question, $pdo)
{
    // TODO
}
?>