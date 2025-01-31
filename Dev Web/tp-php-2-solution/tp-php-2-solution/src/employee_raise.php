<?php
require_once __DIR__ . '/../vendor/autoload.php';
use Acme\Employee;

$employees = array();
$employees[] = new Employee(1, "superman", 1.27, 2012 - 1932);
$employees[] = new Employee(2, "batman", 1.00, 2012 - 1939);
$employees[] = new Employee(3, "spiderman", 0.82, 2012 - 1962);

function employee_raise($e)
{
    if (is_object($e) && $e instanceof Employee) {
        $e->setSalary($e->getSalary() * 1.05);
    } else {
        throw new Exception("Le paramètre n'est pas une instance de Employee\n\n");
    }
}
echo "Avant augmentation :\n";
foreach ($employees as $emp)
    echo $emp;
array_map("employee_raise", $employees);
echo "Après augmentation :\n";
foreach ($employees as $emp)
    echo $emp;

// tester la gestion d'exceptions
try {
    employee_raise(array());
} catch (Exception $e) {
    echo $e->getMessage();
}


/*
 * // gestionnaire d'erreurs utilisateur
 * function myErrorHandler($errno, $errstr, $errfile, $errline) {
 * echo "AN ERROR HAS OCCURRED :\n\t".$errno."\n\t".$errstr."\n\t".$errfile."\n\t".$errline."\n\n";
 * }
 *
 * set_error_handler("myErrorHandler");
 *
 * try {
 * echo $employees[0]->_some_salary;
 * } catch(Exception $e) {
 * echo $e->getMessage();
 * }
 * /*
 */
?>