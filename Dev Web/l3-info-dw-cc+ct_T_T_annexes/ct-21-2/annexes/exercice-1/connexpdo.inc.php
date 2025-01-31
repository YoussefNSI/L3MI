<?php
require("js.php");
function connexpdo($db)
{
    $sgbd = "mysql"; // choix de MySQL
    $host = "localhost";
    $charset = "UTF8";
    $user = "root"; // TODO : user id
    $pass = ""; // TODO : password
    try {
        $pdo = new PDO("$sgbd:host=$host;dbname=$db;charset=$charset", $user, $pass);
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        return $pdo;
    } catch (PDOException $e) {
        displayException($e);
        exit();
    }
}
?>
