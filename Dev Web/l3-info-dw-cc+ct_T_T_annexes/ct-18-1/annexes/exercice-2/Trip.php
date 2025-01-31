<?php
require 'Country.php';

class Trip
{

    // pays d'origine (instance de Country)
    public $origin;

    // pays de destination (instance de Country)
    public $destination;

    // $o et $d sont deux instances de Country
    public function __construct($o, $d)
    {
        $this->origin = $o;
        $this->destination = $d;
    }

    public function getDistance()
    {
        // A COMPLETER
    }
}
?>
