<?php
require 'Ville.php';

$city1 = new Ville();
$city1->setNom("Saumur");
$city1->setRegion("  Pays de la Loire   ");
$city1->setPopulation(27486);
echo $city1;
echo "<br/>";

$city2 = new Ville();
$city2->setNom("Toulon");
$city2->setRegion("     Provence-Alpes-Côte d'Azur  ");
$city2->setPopulation(165514);
$city2->setPrefecture();
echo $city2;
echo "<br/>";

try {
	$city3 = new Ville();
	$city3->setNom("Laval");
	$city3->setRegion("Pays de la Loire  ");
	$city3->setPopulation(121399);
	echo $city3;
} catch (Exception $e) {
	echo $e->getMessage();
}
?>
