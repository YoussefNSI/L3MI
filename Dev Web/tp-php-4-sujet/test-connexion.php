<?php
require ("connexpdo.inc.php");
require_once ("js.php");

try {
    $objdb = connexpdo("voitures");
    var_dump($objdb);
} catch (PDOException $e) {
    echo "erreur";
    displayException($e);
}
?>