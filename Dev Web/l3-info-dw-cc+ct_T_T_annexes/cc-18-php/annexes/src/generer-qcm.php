<?php
require 'connexpdo.inc.php';
require 'selectionner-questions-alternatives.php';

$db = "l3info_cc_18_php_autoqcm";
$pdo = connexpdo($db);

$a = intval($_GET['questions'] ?? 4);
if (! isset($_GET['tirage'])) {
    $a = range(1, $a);
}

$qas = selectionnerQuestionsAlternatives($pdo, $a);
require 'gabarit-qcm.php';
?>