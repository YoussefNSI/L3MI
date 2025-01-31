<?php
require_once('sat.php');

$mars = new Planete(); $mars->setNom("Mars"); $mars->setRayon(3397); $mars->setGravitation(3.69);
$ph = new Satellite(); $ph->setNom("Phobos"); $ph->setPlanete($mars->getNom()); $ph->setRotation(0.32);
$de = new Satellite(); $de->setNom("Deimos"); $de->setPlanete($mars->getNom()); $de->setRotation(1.26);
$mars->addSatellite($ph);
$mars->addSatellite($de);
echo $mars;
?>