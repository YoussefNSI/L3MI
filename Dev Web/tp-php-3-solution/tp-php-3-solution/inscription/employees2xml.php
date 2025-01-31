<?php
require_once ('Employee.php');
// require_once('Manager.php');
/*
 * $e1 = new Employee(1,"superman",20000,2012-1932);
 * $e2 = new Employee(2,"batman",10000,2012-1939);
 * $e3 = new Employee(3,"spiderman",5000,2012-1962);
 * $employees[] = $e1; $employees[] = $e2; $employees[] = $e3;
 * $m = new Manager(4,"wonder woman",3.14,2012-1941);
 * $m->add($e1->getId()); $m->add($e2->getId()); $m->add($e3->getId());
 * $employees[] = $m;
 */
$file = dirname(__FILE__) . "/employees.csv";
$employees = [];
if (! file_exists($file) || ! $id_file = fopen($file, "r")) {
    exit();
} else {
    while ($tab = fgetcsv($id_file, 200, ";")) {
        $id = (int) $tab[0];
        $nom = $tab[1];
        $salaire = (float) $tab[2];
        $age = (int) $tab[3];
        $e = new Employee($id, $nom, $salaire, $age);
        $employees[] = $e;
    }
    fclose($id_file);
}

$file = fopen("employees.xml", "w");
if ($file === false)
    die("could not open file for writing");
fwrite($file, "<?xml version='1.0' encoding='UTF-8'?>\n");
fwrite($file, "<!DOCTYPE employees SYSTEM 'employees.dtd'>\n");
fwrite($file, "<employees>\n");
foreach ($employees as $employee) {
    if (get_class($employee) == "Employee") {
        fwrite($file, "\t<employee>\n");
        toXML($employee);
        fwrite($file, "\t</employee>\n");
    } else if (get_class($employee) == "Manager") {
        fwrite($file, "\t<Manager>\n");
        toXML($employee);
        fwrite($file, "\t<subordinate>\n");
        foreach ($employee->getArrEmployeesId() as $id)
            fwrite($file, "\t\t<refId>$id</refId>\n");
        fwrite($file, "\t</subordinate>\n");
        fwrite($file, "\t</Manager>\n");
    }
}
fwrite($file, "</employees>\n");

function toXML($employee)
{
    global $file;
    fwrite($file, "\t<id>" . $employee->getId() . "</id>\n");
    fwrite($file, "\t<name>" . $employee->getName() . "</name>\n");
    fwrite($file, "\t<salary>" . $employee->getSalary() . "</salary>\n");
    fwrite($file, "\t<age>" . $employee->getAge() . "</age>\n");
}

echo file_get_contents("employees.xml");
?>
