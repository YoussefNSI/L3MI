<?php

class Country
{

    // nom du pays (par ex. "France")
    private $name;

    // code ISO à 2 lettres (par ex. "FR")
    private $code2;

    // prix local du Big Mac (par ex. 3,90)
    private $localPrice;

    // taux de change officiel du $ en monnaie locale (par ex. 0,74)
    private $dollarOfficial;

    // BMI du $ en monnaie locale (par ex. 0,81)
    private $dollarBMI;

    // $code2 est un code ISO à 2 lettres (par ex. "FR" pour la France)
    public function __construct($code2)
    {
        // A COMPLETER
    }

    public function get()
    {
        return [
            "name" => $this->name,
            "code2" => $this->code2,
            "localPrice" => $this->localPrice,
            "dollarOfficial" => $this->dollarOfficial,
            "dollarBMI" => $this->dollarBMI
        ];
    }
}

?>
