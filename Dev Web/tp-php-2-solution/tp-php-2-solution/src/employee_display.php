<?php

// lève une exception si erreur de typage à l'appel de fonctions/méthodes
// 1 déclaration par fichier incluant Employee.php !
declare(strict_types = 1);
require_once __DIR__ . '/../vendor/autoload.php';
use Acme\IEmployee;
use Acme\Employee;

$employees = array();
$employees[] = new Employee(1, "superman", 1.27, 2012 - 1932);
$employees[] = new Employee(2, "batman", 1.00, 2012 - 1939);
$employees[] = new Employee(3, "spiderman", 0.82, 2012 - 1962);
print_r($employees);
foreach ($employees as $emp)
    echo $emp . "\n";

// 3 méthodes d'affichage possibles avec fonction de rappel (call-backs)
// option 1 - use closure (anonymous function)
$closure = function (IEmployee $e): float {
    return $e->getSalary();
};
$salaries = array_map($closure, $employees);
print_r($salaries);

// option 2 : use external getter
function getEmployeeSalary(IEmployee $emp): float
{
    return $emp->getSalary();
}
$salaries = array_map("getEmployeeSalary", $employees);
print_r($salaries);

// option 3 : use static class method
$salaries = array_map(Employee::class . "::sGetSalary", $employees); // !! usage du FQCN est requis : 'Acme\Employee' fonctionnera mais pas 'Employee'
print_r($salaries);

// Salaire moyen
echo "mean salary = " . array_sum($salaries) / count($salaries) . "\n";

// Autre solution utilisant une fermeture liée à une variable non-locale (syntaxe `use`) et mutables (syntaxe `&`)
function salaireTotal()
{
    global $employees;
    $t = 0;
    array_walk($employees, function ($e) use (&$t) {
        $t += $e->getSalary();
    });
    return $t;
}

echo "Salaire moyen : " . salaireTotal() / count($employees);
?>