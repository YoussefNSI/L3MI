<?php
require 'population.php';

$population = new Population();
$population->importerJSON("population.json");
$population->afficher();
?>
