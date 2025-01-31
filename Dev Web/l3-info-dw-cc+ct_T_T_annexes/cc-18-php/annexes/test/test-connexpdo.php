<?php
require __DIR__ . '/../src/connexpdo.inc.php';

$db = "l3info_cc_18_php_autoqcm";
$pdo = connexpdo($db);
var_dump($pdo);
?>