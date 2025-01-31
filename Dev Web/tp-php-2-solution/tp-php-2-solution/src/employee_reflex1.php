<?php
require_once __DIR__ . '/../vendor/autoload.php';
use Acme\Employee;

$e = new Employee(0,"euler",2.718,2012-1707);
echo "**Classe :\n";
echo get_class($e)."\n";
echo "**Classe parente :\n";
if(get_parent_class($e)===false)
    echo "Pas de classe parente \n";
else 
    echo get_parent_class($e)."\n";
    

// tableau des valeurs par défaut (si définies) des propriétés statiques et non-statiques
// de la classe qui sont accessibles dans le contexte d'appel
echo "** Propriétés visibles ayant une valeur par défaut :\n";
print_r(get_class_vars(Employee::class));

// tableau des propriétés non-statiques d'un objet qui sont accessibles
// dans le contexte d'appel
echo "** Propriétés publiques :\n";
print_r(get_object_vars($e));

// appelle get_object_vars depuis l'objet
echo "** Toutes les propriétés :\n";
$e->introspection();
?>