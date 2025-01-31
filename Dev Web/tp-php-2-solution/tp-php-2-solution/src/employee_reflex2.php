<?php
require_once __DIR__ . '/../vendor/autoload.php';
use Acme\Employee;

$e = new Employee(0,"euler",2.718,2012-1707);
$reflector = new ReflectionClass($e);
// alternative
$reflector = new ReflectionClass(Employee::class); // !! usage du FQCN est requis : 'Acme\Employee' fonctionnera mais pas 'Employee'
echo $reflector->getName()."\n";
echo $reflector->getParentClass()."\n";
print_r($reflector->getProperties());
print_r($reflector->getMethods());
echo "\n";

/*
// alternative à classe Employee avec deux variantes de __get :
// - l'une sans contrôle d'accès aux champs (champs non-publiques accessibles)
// - l'autre (__getStrict) avec contrôle d'accès en utilisant ReflectionClass
class Employee {
	private $id;
	public $name;
	protected $salary;
	private $age;

	function __construct($id,$name,$salary,$age) {
		if(!empty($id) && !is_int($id))
			throw new Exception("incorrect employee id");
		$this->name = $name;
		if(!empty($name) && (!is_string($name) || is_null($name)))
			throw new Exception("incorrect employee name");
		$this->salary = $salary;
		if(!empty($salary) && !is_float($salary))
			throw new Exception("incorrect employee salary");
		$this->age = $age;
		if(!empty($age) && (!is_int($age) || $age<0))
			throw new Exception("incorrect employee age");
		
		$this->id = $id;
		$this->name = $name;
		$this->salary = $salary;
		$this->age = $age;
	}

	// using 'get_object_vars' - no checks on access rights
	function __get($ppy) {
		$ppys = get_object_vars($this);
		//print_r($ppys); echo $ppy." - ".$ppys["name"];
		if(array_key_exists($ppy,$ppys)) {
			if(isset($ppys[$ppy])) {
				return $ppys[$ppy];
			} else {
				echo "__get(string): warning - unset property $ppy";
				return null;
			}
		} else {
			throw new Exception("__get(string): wrong input property name '$ppy'");
		}
	}
	// using ReflectionClass - proper checks on access rights
	function __getStrict($ppy) {
		$reflector = new ReflectionClass("Employee");
		if($reflector->hasProperty($ppy)) {
			$rppy = $reflector->getProperty($ppy);
			if($rppy->isStatic()) {
				throw new Exception(__METHOD__.": requesting static property");
			} else {
				if($rppy->isPublic()) {
					return $rppy->getValue($this);
				} else {
					// TODO: should not throw based on call context: $this for private, subclass for $protected
					throw new Exception(__METHOD__.": requesting {$rppy}->getModifiers()}");
				}
			}
		} else {
			throw new Exception(__METHOD__.": wrong property name");
		}
	}
	// using get_object_vars - no checks on access rights
	public function __set($ppy,$val) {
		$ppys = get_object_vars($this);
		if(array_key_exists($ppy,$ppys)) {
			// partial type checking of input value
			if(!isset($this->$ppy) || getType($this->$ppy)==getType($val)) {
				$this->$ppy = $val;
			} else {
				throw new Exception("__set: wrong input value type");
			}
		} else {
			throw new Exception("__set: wrong input property name");
		}
	}
	function __toString() {
		return "employee: id=$this->id name=$this->name salary=$this->salary age=$this->age\n";
	}
}

$e = new Employee(0,"euler",2.718,2012-1707);

// ok : appel à __get
echo $e->age."\n";
// ok : appel à __set
$e->age = 12;
// ok : appel explicite à __get
echo $e->__get("age")."\n";
// lance une exception
echo $e->__getStrict("age")."\n";
/**/
?>