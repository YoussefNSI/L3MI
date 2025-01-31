<?php
require_once 'js.php';

function connexpdo(string $db)
{
    $sgbd = "mysql"; // choix de MySQL (fonctionnera aussi avec MariaDB !)
    $host = "localhost";
    $port = 3306; // port par défaut de MySQL (à adapter selon votre config et votre choix entre mysql/mariadb)
    $charset = "UTF8";
    $user = "etudiant"; // user id
    $pass = "antietud"; // password

    try {
        $pdo = new pdo("$sgbd:host=$host;port=$port;dbname=$db;charset=$charset", $user, $pass, array(
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION
        ));
        return $pdo;
    } catch (PDOException $e) {
        displayException($e);
        exit();
    }
}
?>