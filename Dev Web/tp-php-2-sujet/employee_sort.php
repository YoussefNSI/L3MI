<?php

require_once 'employee.php';
require_once 'employee_display.php';
require_once 'employee_raise.php';

usort($employees, function($a, $b) {
    return $a->getSalary() <=> $b->getSalary();
});

print_r($employees);