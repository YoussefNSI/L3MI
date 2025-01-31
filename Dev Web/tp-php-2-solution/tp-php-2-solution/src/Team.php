<?php
declare(strict_types=1); // lève une exception si erreur de typage à l'appel de fonctions/méthodes
namespace Acme;
require_once __DIR__ . '/../vendor/autoload.php';

class Team {
	protected $arrEmployees;	
	public function __construct() {
		$this->arrEmployees=array();
	}	
	public function add(Employee $employee) {
	    $this->arrEmployees[$employee->getId()]=$employee;
	}	
	public function __toString() : string {
		$arr=array();
		foreach($this->arrEmployees as $employee) {
			if (get_class($employee)=='Acme\Employee') {
				$arr[]= $employee->__toString();
			} elseif (get_class($employee)=='Acme\Manager' ) {
			    $s = $employee->__toString() . "subordinates=[";
				foreach($employee->getArrEmployeesId() as $id) {
					$s .= $this->arrEmployees[$id]->getName()." ";
				}
				$s .= "]\n";
				$arr[]= $s;
			} else {
			    throw new \Exception("erreur de type");
			}
		}
		return implode("\n", $arr);
	}
}
?>