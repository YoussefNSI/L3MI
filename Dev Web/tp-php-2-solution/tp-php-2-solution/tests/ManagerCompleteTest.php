<?php
// namespace Tests; // optional - for cleanly spearating test classes from target classes namespace-wise: in any case, `./vendor/bin/phpunit tests` will search ./tests for *Test.php classes
require_once __DIR__ . '/../vendor/autoload.php';
use Acme\Manager;
use Acme\Employee;
use PHPUnit\Framework\TestCase;

final class ManagerCompleteTest extends TestCase {
	private $manager;
	
	public function setUp(): void {
		$this->manager = new Manager(0, "John", 2000.0, 40);
	}

	public function tearDown(): void {
		unset($this->manager);
	}
	
	public function testSetId(): void {
		$id = 10;
		$this->manager->setId($id);
		$this->assertEquals($this->manager->getId(),10);
	}
	
	public function testSetName(): void {
		$name = "Alfred";
		$this->manager->setName($name);
		$this->assertEquals($this->manager->getName(),$name);
	}
	
	public function testSetSalary(): void {
		$salary = 30;
		$this->manager->setSalary($salary);
		$this->assertEquals($this->manager->getSalary(),$salary);
	}
	
	public function testSetAge(): void {
		$age = 22;
		$this->manager->setAge($age);
		$this->assertEquals($this->manager->getAge(),$age);
	}
	
	public function testAddEmployee(): void {
		$employees[]= new Employee(2, "Pif", 10, 10);
		$employees[] = new Employee(0, "Paf", 100, 100);
		$employees[] = new Employee(1, "Pouf", 5, 50);
		$this->manager->add(2);
		$this->manager->add(0);
		$this->manager->add(1);
		$this->assertEquals($this->manager->getArrEmployeesId(),
							array_map(function($e) {return $e->getId();},
									  $employees));
	}
}
?>
