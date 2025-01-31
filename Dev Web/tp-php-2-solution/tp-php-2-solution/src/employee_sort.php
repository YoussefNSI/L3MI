<?php
require_once __DIR__ . '/../vendor/autoload.php';
use Acme\Employee;

$employees = array();
$employees["super"] = new Employee(1,"superman",1.27,2012-1932);
$employees["bat"] = new Employee(2,"batman",1.00,2012-1939);
$employees["spider"] = new Employee(3,"spiderman",0.82,2012-1962);

//print_r($employees);

function my_sort(Employee $e1, Employee $e2) : int {
	return $e1->getSalary() <=> $e2->getSalary();
	/* Equivalent à :
	if($e1->getSalary()<$e2->getSalary())
		return -1;
	else
		if($e1->getSalary()==$e2->getSalary())
			return 0;
		else
			return 1;
			*/
}

echo "\nKey-preserving salary-increasing sorting\n";
uasort($employees,"my_sort");
print_r($employees);
//foreach($employees as $e) { echo $e->getName()."\n"; }

echo "\nNon-key-preserving salary-increasing sorting\n";
usort($employees,"my_sort");
print_r($employees);
//foreach($employees as $e) { echo $e->getName()."\n"; }
?>