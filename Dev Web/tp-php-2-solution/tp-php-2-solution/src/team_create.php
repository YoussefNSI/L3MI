<?php
declare(strict_types=1); // lève une exception si erreur de typage à l'appel de fonctions/méthodes
require_once __DIR__ . '/../vendor/autoload.php';

use Acme\Team;
use Acme\Employee;
use Acme\Manager;


$team=new Team();
$e1 = new Employee(1,"superman",1.27,2012-1932);
$e2 = new Employee(2,"batman",1.00,2012-1939);
$e3 = new Employee(3,"spiderman",0.82,2012-1962);
$team->add($e1);
$team->add($e2);
$team->add($e3);
$m = new Manager(4,"wonder woman",3.14,2012-1941);
$m->add($e1->getId());
$m->add($e2->getId());
$m->add($e3->getId());
$team->add($m);

echo $team;
?>