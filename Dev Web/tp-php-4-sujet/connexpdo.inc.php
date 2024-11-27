<?php
function connexpdo(string $db){
    $sgbd = "mysql";
    $host = "localhost";
    $port = 3306;
    $charset = "UTF8";
    $user = "root";
    $pass = "gRBvbZ";
    $pdo = new pdo("$sgbd:host=$host;port=$port;charset=$charset;dbname=$db", $user, $pass,
    array(PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION));
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    return $pdo;
}
