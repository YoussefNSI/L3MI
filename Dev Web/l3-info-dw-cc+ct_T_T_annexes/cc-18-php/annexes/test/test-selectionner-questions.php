<?php
require __DIR__ . '/../src/connexpdo.inc.php';
require __DIR__ . '/../src/selectionner-questions.php';

function testSelectionnerQuestions($m, $pdo, $a)
{
    try {
        echo "\n================\n", $m, " :\n";
        $questions = selectionnerQuestions($pdo, $a);
        print_r($questions);
    } catch (InvalidArgumentException $e) {
        echo $e->getMessage();
    }
}

$db = "l3_cc_qcm";
$pdo = connexpdo($db);
testSelectionnerQuestions("Tirage aléatoire de 2 questions", $pdo, 2);
testSelectionnerQuestions("Sélection des questions 11 et 26", $pdo, [11,26]);
testSelectionnerQuestions("Levée d'exception (1)", $pdo,"bad");
testSelectionnerQuestions("Levée d'exception (2)", $pdo,-30);
testSelectionnerQuestions("Levée d'exception (3)", $pdo,100);
testSelectionnerQuestions("Levée d'exception (4)", $pdo,[100]);
testSelectionnerQuestions("Levée d'exception (5)", $pdo,["bad"]);
?>