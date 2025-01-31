<?php

/**
 * Lit une partie de RushHour stockée dans un fichier CSV et la convertit en tableau assocatif.
 * 
 * Le fichier CSV contient l'historique des mouvements effectués lors d'une partie.
 * Chaque ligne correspond à un mouvement de voiture :
 * - le premier champ est l'identifiant de la voiture 
 * - le second est le décalage effectué.
 * 
 * Les champs sont séparés par des virgules	et les lignes sont triées dans l'ordre chronologique des mouvements.
 * Le tableau retourné par la fonction doit respecter l'ordre des mouvements et suivre le format suivant :
 *     Array
 *    (
 *         [0] => Array
 *             (
 *                 [voiture] => 1
 *                 [decalage] => -1
 *             )
 * 
 *         [1] => Array
 *             (
 *                 [voiture] => 5
 *                 [decalage] => -3
 *             )
 *         ...
 *     )
 * 
 *  @param  $filename Nom du fichier à lire.
 *  @return Un tableau indexé de tableaux associatifs.
 *          En cas d'erreur, retourne NULL.
 **/
function lire_partie( string $filename ) : array
    {
    }

?>