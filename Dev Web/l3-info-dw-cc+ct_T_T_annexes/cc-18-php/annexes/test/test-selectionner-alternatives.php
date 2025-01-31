<?php
require __DIR__ . '/../src/connexpdo.inc.php';
require __DIR__ . '/../src/selectionner-alternatives.php';

$db = "l3info_cc_18_php_autoqcm";
$pdo = connexpdo($db);
$alternatives = selectionnerAlternatives(11, $pdo);
print_r($alternatives);
?>