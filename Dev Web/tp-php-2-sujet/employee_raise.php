<?php

require_once 'employee_display.php';
require_once 'employee.php';

echo "Après augmentation de 5%: <br>";
foreach ($employees as $employee) {
    if (!($employee instanceof Employee)) {
        throw new Exception("Le paramètre n'est pas une instance de la classe Employee.");
    }
    else{
        $employee->setSalary($employee->getSalary() * 1.05);
        echo $employee . "<br>";
    }
    
}