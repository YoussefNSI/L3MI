<?php

/**
 * Lit une table à partir d'un fichier XML.
 * 
 * Par exemple, l'instruction
 *      lire_xml_table("l3_cc_24_php_rush_hour.xml", "DEFI");
 * renvoie :
 *      Array
 *      (
 *          [0] => Array(
 *              [defi_id] => 1
 *              [nr_cases] => 6
 *              [nr_voitures] => 4
 *          )
 *          [1] => Array(
 *              [defi_id] => 2
 *              [nr_cases] => 6
 *              [nr_voitures] => 6
 *          )
 *          [2] => Array(
 *              [defi_id] => 3
 *              [nr_cases] => 6
 *              [nr_voitures] => 4
 *          )
 *      )
 * 
 * @param $filename  nom du fichier XML à lire
 * @param $tablename nom de la table à extraire
 * @return array
 */
function lire_xml_table( string $filename, string $tablename ) : array
    {
    }

/**
 * Calcule la durée moyenne des parties.
 * 
 * @param  $xmlname  nom du fichier XML
 * @return float    la durée moyenne des parties
 */
function s1_temp_moyen_parties( string $xmlname ) : float
    {
    }

/**
 * Calcule le nombre minimum et maximum de mouvements de la
 * voiture rouge pour les parties jouées.
 * 
 * Exemple de tableau à retourner :
 * Array
 * (
 *     [mouvements_min] => 6
 *     [mouvements_max] => 3
 * )
 * 
 * @param  $xmlname  nom du fichier XML
 * @return array     tableau associatif avec les valeurs min et max
 */
function s2_mouvements_voiture( string $xmlname ) : array
    {
    }

/**
 * Récapitule les statistiques du jeu calculées par les fonctions précédentes.
 * Exemple :
 * Array
 * (
 *     [nb_parties] => 10
 *     [temps_moyen_parties] => 18.5
 *     [voiture_rouge_min_mouvs] => 6
 *     [voiture_rouge_max_mouvs] => 3
 * ) 
 * @param  $xmlname nom du fichier XML
 * @return array tableau associatif contenant les statistiques
 * 
 */
function stats_jeu( string $xmlname ) : array
    {
    }

// Tests :
$filename = "l3_cc_24_php_rush_hour.xml";

$defi = lire_xml_table( $filename, "DEFI" );
echo "<pre>défis :<br>";
print_r( $defi );
echo "</pre>";

echo "s1 : " . s1_temp_moyen_parties( $filename );

$s2 = s2_mouvements_voiture( $filename );
echo "<br><pre>s2 :<br>";
print_r( $s2 );
echo "</pre>";

echo "<br><pre>stats jeu :<br>";
$stats_jeu = stats_jeu( $filename );
print_r( $stats_jeu );
echo "</pre>";
?>