<?php
require "./connexpdo.inc.php";
// crée ou re-démarre une session 
session_start();

// Q5 Récupérer les données communiquées en HTTP POST :
// (a) id de la voiture à déplacer: `voiture_id`
// (b) décalage à effectuer de valeur -1 ou 1 : `decalage` 
// (c) listing (de clé `voitures`) des voitures du défi sous la forme d'un objet JSON où 
// - chaque clé est l'id d'une voiture
// - `taille` correspond à la taille de la voiture
// - `orientation` correspond à l'orientation de la voiture ("H" pour "horizontale", "V" pour "verticale")
// Exemple :
// {
//      "2" => {"taille":2, "orientation":'V'},
//      "14" => {"taille":3, "orientation":'H'},
//      ...
// }
$voiture_id = null;
$decalage   = null;
$voitures   = null;

// Q6 Extraire des données de session l'id de la partie, l'id du défi, la taille de ligne et le nombre de voitures.
$partie_id   = null;
$defi_id     = null;
$nr_cases    = null;
$nr_voitures = null;

// Q7 A partir des données de session contenant les coordonnées (x,y) de la case de référence de chaque voiture,
// créer un tableau $positions contenant pour chaque voiture le tableau des coordonnées de toutes les cases qu'elle occupe
// selon l'orientation et la taille communiquées 
// Exemple : ["1" => [["x"=>3, "y"=>2], ["x"=>3, y=>"3]], "5" => [["x"=>4, "y"=>3], ...], ...]
$positions = [];

// Q8. Le mouvement est invalide si la voiture sort du plateau ou si elle chevauche une autre voiture.
// Vérifier la validité du mouvement (stocker le résultat dans un booléen) et, s'il est valide, 
// mettre à jour les coordonnées des cases de référence des voitures dans les données de session.
// Noter que seule la première case ou bien la dernière case de la voiture peuvent être en conflit selon le décalage demandé.
// Il suffit donc de tester le chevauchement de la nouvelle case de référence de la voiture avec toutes les cases occupées 
// par toutes les voitures dans l'état courant.
// A cet effet, vous pouvez utiliser le tableau créé en question précédente et la fonction libre().

// Renvoie `true` si les coordonnées ($x,$y) ne figurent pas dans le tableau $positions
function libre( $x, $y )
    {
    global $positions;
    foreach ( $positions as $allxy ) {
        foreach ( $allxy as $xy ) {
            if ( $xy[ "x" ] === $x && $xy[ "y" ] === $y ) {
                return false;
                }
            }
        }
    return true;
    }

$valide = false;


// Q9 Si le mouvement est invalide, renvoyer le tableau PHP ["statut"=>"KO","debug"=>"..."] formatté en objet JSON


// Si le mouvement est valide, il est inséré en base de données
$pdo   = connexpdo( "l3_cc_24_php_rush_hour" );
$query = "INSERT INTO MOUVEMENT (partie_id, voiture_id, decalage) VALUES (:partie_id, :voiture_id, :decalage)";
$stt   = $pdo->prepare( $query );
$data  = [ ":partie_id" => intval( $partie_id ), ":voiture_id" => intval( $voiture_id ), ":decalage" => intval( $decalage ) ];
$stt->execute( $data );

// Q10 Si le mouvement est valide, renvoyer le même type d'objet JSON qu'en question 3 (objet contenant notamment les coordonnées des cases de référence des voitures) :
// - si le mouvement conclut la partie, fixer la propriété "statut" de la réponse à la valeur "VICTOIRE" et mettre à jour la date de fin de la partie en base de données
// - sinon, la fixer simplement à la valeur "OK"
?>