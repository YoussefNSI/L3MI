<?php
require_once 'js.php';

function connexpdo(string $db)
{
    $sgbd = "mysql";
    $host = "localhost";
    $port = 3306;
    $charset = "UTF8";
    $user = "";
    $pass = "";

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
