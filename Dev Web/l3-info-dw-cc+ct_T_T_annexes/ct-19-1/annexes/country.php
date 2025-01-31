<?php
require "connexpdo.inc.php";

$dbname = "l3info_ct_19_1_pays";
$xmlfile = "./data/country-codes.xml";

// Q1-1 extraire et convertir la table COUNTRY(ID,NAME) en un tableau associatif
// [1 => "Afghanistan", ...] où la clé d'un pays est la valeur de son champ ID


// Q2-1 extraire et convertir les codes du fichier XML en un tableau de tableaux associatifs
// [..., ["country" => "France", "abbreviation" => "FR"] ,...]


// Q2-2 produire le fichier JS "./src/country-codes.js"

// Q3-1 récupérer les couples pays-continents de la BDD sous la forme d'un tableau de tableaux associatifs
// [..., ["country" => "France", "continent" => "Europe"] ,...]

// Q3-2 produire le fichier JS "./src/country-continents.js"

require "gabarit.php";
?>