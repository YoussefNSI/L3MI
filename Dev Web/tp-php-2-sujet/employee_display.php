<?php

require_once 'employee.php';

$employees = [
    new Employee(1, 'John Doe', 50000, 30),
    new Employee(2, 'Jane Smith', 60000, 35),
    new Employee(3, 'Alice Johnson', 55000, 28)
];

$somme_salaire = 0;
foreach ($employees as $employee) {
    echo $employee . "<br>";
    $somme_salaire += $employee->getSalary();
}



echo "mean salary = " . $somme_salaire / count($employees) . "<br>";