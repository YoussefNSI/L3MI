<?php
$expectCont = array ( 'Asia' => 67.2, 'Europe' => 75.2, 'Africa' => 52.6, 'Oceania' => 69.8, 'North America' => 72.8, 'South America' => 70.9, ) ;

if(__FILE__ == $_SERVER["SCRIPT_FILENAME"]){
    echo "<pre>";
    print_r($expectCont);
    echo "</pre>";
}
?>