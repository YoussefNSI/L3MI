<?php
header( "Content-Type: application/json; charset=UTF-8" );

session_start();

if ( ! key_exists( "file", $_SESSION ) ) {
    $filename = "./data/communes.csv";
    $csv      = file_get_contents( $filename );
    $clés     = [ "CDC", "CHEFLIEU", "REG", "DEP", "COM", "AR", "CT", "TNCC", "ARTMAJ", "NCC", "ARTMIN", "NCCENR" ];
    $clés_f   = array_flip( $clés );
    $communes = [];

    $row = 1;
    if ( ( $handle = fopen( $filename, "r" ) ) !== FALSE ) {
        while ( ( $data = fgetcsv( $handle, 1000, ";" ) ) !== FALSE ) {
            if ( $row !== 1 ) {
                $num     = count( $data );
                $commune = [];
                for ( $c = 0; $c < $num; $c++ ) {
                    if ( $c === $clés_f[ "REG" ] || $c === $clés_f[ "DEP" ] || $c === $clés_f[ "NCCENR" ] ) {
                        $commune[ $clés[ $c ] ] = $data[ $c ];
                        }
                    }
                $communes[]         = $commune;
                $_SESSION[ "file" ] = $communes;
                }
            else {
                ++$row;
                }
            }
        fclose( $handle );
        }
    }
else {
    $communes = $_SESSION[ "file" ];
    }

$clé_région = "région";
$clé_bloc   = "bloc";

if ( ! key_exists( $clé_région, $_POST ) ) {
    throw new Exception( "pas de clé 'région' communiquée" );
    }
if ( ! key_exists( $clé_bloc, $_POST ) ) {
    throw new Exception( "pas de clé 'bloc' communiquée" );
    }

$région          = $_POST[ $clé_région ];
$bloc            = (int) $_POST[ $clé_bloc ];
$lignes          = 100;
$start           = ( $bloc - 1 ) * $lignes;
$end             = $lignes;
$communes_région =
    array_filter(
        $communes,
        // fn ($commune) => $commune[ "DEP" ] === "49"
        function ($commune) use ($région) {
            return $commune[ "REG" ] === $région; }
    );
$communes_région =
    array_map(
        function ($commune)
            {
            [ "région" => $commune[ "REG" ], "département" => $commune[ "DEP" ], "nom" => $commune[ "NCCENR" ] ];
            },
        $communes_région
    );
$communes_région = array_slice( array_values( $communes_région ), $start, $end );
echo json_encode( $communes_région );
/**/
?>