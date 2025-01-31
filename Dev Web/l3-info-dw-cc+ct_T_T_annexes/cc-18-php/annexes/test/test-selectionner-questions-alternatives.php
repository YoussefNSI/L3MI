<?php
require __DIR__ . '/../src/connexpdo.inc.php';
require __DIR__ . '/../src/selectionner-questions-alternatives.php';

$db = "l3info_cc_18_php_autoqcm";
$pdo = connexpdo($db);

echo "\nTirage aléatoire de 2 questions :\n";
$questions = selectionnerQuestionsAlternatives($pdo,2);
print_r($questions);

echo "\nTirage aléatoire de 1 question :\n";
$questions = selectionnerQuestionsAlternatives($pdo,1);
print_r($questions);

echo "\nSélection des questions 3 et 5 :\n";
$questions = selectionnerQuestionsAlternatives($pdo, [3,5]);
print_r($questions);
?>