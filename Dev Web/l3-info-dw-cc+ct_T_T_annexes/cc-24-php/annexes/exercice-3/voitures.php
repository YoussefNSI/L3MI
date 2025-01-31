<?php
require "./connexpdo.inc.php";

$pdo      = connexpdo( "l3_cc_24_php_rush_hour" );
$query    = "SELECT * FROM VOITURE";
$stt      = $pdo->query( $query );
$autos    = $stt->fetchAll( PDO::FETCH_ASSOC );
$voitures = [];
foreach ( $autos as $auto ) {
    $voitures[ $auto[ "voiture_id" ] ] = $auto;
    unset( $voitures[ $auto[ "voiture_id" ] ][ "voiture_id" ] );
    }

echo json_encode( $voitures );
?>