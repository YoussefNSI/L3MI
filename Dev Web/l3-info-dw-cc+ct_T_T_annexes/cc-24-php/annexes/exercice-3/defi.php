<?php
require "./connexpdo.inc.php";
// crée ou re-démarre une session 
session_start();
// détruit les données de session
session_unset();

// Q1 Récupérer les champs du formulaire en HTTP GET : la taille de ligne et le nombre de voitures
$nr_cases    = 6;
$nr_voitures = 4;

// Extrait le défi correspondant aux champs communiqués
$pdo   = connexpdo( "l3_cc_24_php_rush_hour" );
$query = "SELECT * FROM DEFI WHERE nr_cases = :nr_cases AND nr_voitures = :nr_voitures";
$stt   = $pdo->prepare( $query );
$data  = [ ":nr_cases" => $nr_cases, ":nr_voitures" => $nr_voitures ];
$stt->execute( $data );
$defis = $stt->fetchAll( PDO::FETCH_ASSOC );

// Q2 Si aucun défi ne correspond, renvoyer le tableau PHP ["statut"=>"KO", "defi"=>[], "positions"=>[], "debug"=>"..."] formatté en objet JSON 

// Sinon
// Tire un défi aléatoirement dans la liste des défis conformes
$defi = $defis[ array_rand( $defis, 1 ) ];

// Insère une nouvelle partie dans la base de données à partir du défi sélectionné et de l'heure courante
$query = "INSERT INTO PARTIE (defi_id, debut) VALUES (:defi_id, :debut)";
$stt   = $pdo->prepare( $query );
$debut = date( 'Y-m-d H:i:s' );
$data  = [ ":defi_id" => $defi[ "defi_id" ], ":debut" => $debut ];
$stt->execute( $data );
$partie_id = $pdo->lastInsertId();

// Extrait les positions de départ des voitures pour le défi considéré
$query = "SELECT voiture_id, position FROM POSITION WHERE defi_id = :defi_id";
$stt   = $pdo->prepare( $query );
$data  = [ ":defi_id" => $defi[ "defi_id" ] ];
$stt->execute( $data );
$positions   = $stt->fetchAll( PDO::FETCH_ASSOC );
$xypositions = [];

// Q3 Renvoyer un tableau PHP formatté en objet JSON (exemple) :
// [
//      "statut" => "OK",
//      "defi" => ["defi_id"=>1, "nr_cases"=>6, "nr_voitures"=>4],
//      "positions" => ["1" => ["x"=>3, "y"=>2], "5" => ["x"=>4, "y"=>3], ...],
//      "debug"=> "..."
// ]
// où chaque élément du tableau de clé `positions` a pour clé l'id d'une voiture et pour valeur un tableau associatif
// stockant les coordonnées de la case de référence de la voiture :
// - `x` dénote le numéro de ligne (>=1) de la case
// - `y` dénote le numéro de colonne (>=1) de la case

// Q4 Sauvegarder dans les données de session l'id de la partie créée, l'id du défi, la taille de ligne et le nombre de voitures, 
// et les coordonnées des cases de référence des voitures

?>