<?php 
require 'Trip.php';

$trip = new Trip(new Country("FR"), new Country("CN")); 
var_dump($trip->getDistance());

?>
