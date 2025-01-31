<?php
require 'population.php';

$population = new Population();
$population->importerXML("population.xml");
$population->afficher();
?>
