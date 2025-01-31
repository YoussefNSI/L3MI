<?php
require "ARS.php";

// Questions 1 et 2
$ars1 = new ARS();
echo $ars1;
echo "<br/>";

// Question 3
$p = new Patient("Condriaque", "Hypo", 20, 1, "2021-06-23", "CHU Tours");
$ars1->addPatient($p);
echo $ars1;
echo "<br/>";

// Question 4
$ars1->addFromFile("patients.json");
echo $ars1;
echo "<br/>";

// Question 5
$ars1->sort("nomprenom","age");
echo $ars1;
echo "<br/>";

$ars1->sort("hopital","severite");
echo $ars1;
echo "<br/>";
?>