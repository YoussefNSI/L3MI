<?php
require_once ('Employee.php');

function createEmployee(SimpleXMLElement $element): Employee
{
    // ATTENTION : transtyper les objets SimpleXMLElement
    $id = (int) $element->id;
    $name = (string) $element->name;
    $salary = (int) $element->salary;
    $age = (int) $element->age;
    return new Employee($id, $name, $salary, $age);
}

$filename = "employees.xml";
$xml = simplexml_load_file($filename);
$employee = null;

// Approche 1 : force brute SimpleXML
foreach ($xml->employee as $employeeElement) {
    if (((int) $employeeElement->id) == (int) ($_GET["id"])) {
        $employee = createEmployee($employeeElement);
        break;
    }
}

/*
// Approche 2 : XPath
$qry = "//employee[id={$_GET["id"]}]";

// 2.1 : XPath + SimpleXML
$elts = $xml->xpath($qry);
$employee = createEmployee($elts[0]);

// 2.2 XPath + DOMDocument
$doc = new DOMDocument();
$doc->load($filename);
$xpath = new DOMXPath($doc);
$elements = $xpath->query($qry);
$element = $elements->item(0);
$id = (int) $element->getElementsByTagName("id")->item(0)->textContent;
$name = (string) $element->getElementsByTagName("name")->item(0)->textContent;
$salary = (int) $element->getElementsByTagName("salary")->item(0)->textContent;
$age = (int) $element->getElementsByTagName("age")->item(0)->textContent;
$employee = new Employee($id, $name, $salary, $age);
/** */

/*
 * Un encodage JSON de toutes les propriétés de l'instance Employee suppose
 * que la classe Employee implémente l'interface JsonSerializable.
 * echo json_encode($employee);
 */
echo $employee;
?>
