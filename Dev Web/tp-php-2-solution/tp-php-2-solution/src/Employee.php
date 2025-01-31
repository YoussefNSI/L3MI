<?php
//declare(encoding='UTF-8');
declare(strict_types=1); // lève une exception si erreur de typage à l'appel de fonctions/méthodes
namespace Acme;
require_once __DIR__ . '/../vendor/autoload.php';

class Employee implements IEmployee {
	private $id;
	public $name = "anonymous";
	protected $salary;
	private $age;
	public 	function __construct(int $id, string $name,float $salary, int $age) {
		$this->id = $id;
		$this->name = $name;
		$this->salary = $salary;
		$this->age = $age;
	}
	public function getId() :int { return $this->id; }
	public function getName() :string { return $this->name; }
	public function getSalary() :float { return $this->salary;	}
	public function getAge() :int { return $this->age; }
	public function setId(int $id): void { $this->id=$id; }
	public function setName(string $name): void { $this->name=$name; }
	public function setSalary(float $salary): void { $this->salary=$salary; }
	public function setAge(int $age): void { $this->age=$age; }
	public function __toString() : string {
		return "employee: id=$this->id name=$this->name salary=$this->salary age=$this->age\n";
	}
	// pour illustrer les différences de comportement de
	// get_object_vars selon le contexte d'appel
	function introspection(): void {
		$fields = get_object_vars($this);
		print_r($fields);
	}
	// pour illustrer l'usage de méthode statique
	// comme méthode de rappel dans array_map, etc
	static function sGetSalary(IEmployee $e) : float {
		return $e->getSalary();
	}
}
?>