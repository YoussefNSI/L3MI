<?php
require "connexpdo.inc.php";
$dbname = "l3info_ct_19_1_pays";

// Q4 : récupérer le nom du pays soumis par GET ou POST,
// extraire la durée de vie et type de gouvernement correspondants de la BDD
// et les renvoyer sous la forme d'un objet JSON
// Exemple de réponse pour soumission "country_name=France" :
// {"expectancy":"78.8", "government":"Republic"}
?>