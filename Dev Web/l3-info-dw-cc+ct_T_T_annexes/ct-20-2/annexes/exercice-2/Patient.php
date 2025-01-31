<?php

class Patient {
    public $nomprenom;
	public $age;
	public $severite;
    public $date;
	public $hopital;

    public function __construct(string $nom, string $prenom, int $age, int $severite, string $date, string $hopital) {
        // A COMPLETER
    }

    public function __toString() {
        // A COMPLETER      
    }
}

?>