<?php

// Connexion à la BdD
function connexpdo(string $db)
{
    $sgbd = "mysql"; // choix de MySQL
    $host = "localhost";
    $charset = "UTF8";
    $user = "etudiant"; // user id
    $pass = "antietud"; // password
    try {
        $pdo = new PDO("$sgbd:host=$host;dbname=$db;charset=$charset", $user, $pass);
        // force le lancement d'exception en cas d'erreurs d'exécution de requêtes SQL
        // via eg. $pdo->query()
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        printSuccess("Connexion à la base OK");
        return $pdo;
    } catch (PDOException $e) {
        $msg = "!!! ERREUR !!!<br/>Fichier : ".$e->getFile()."<br/>Ligne : ".$e->getLine()."<br/>Message : ".$e->getMessage()."<br/>Code : ".$e->getCode();
        printError($msg);
        exit();
    }
}

function printError(string $msg){
    echo "<div class=\"error\">".$msg."</div>";
}


function printSuccess(string $msg){
    echo "<div class=\"success\">".$msg."</div>";
}
?>
