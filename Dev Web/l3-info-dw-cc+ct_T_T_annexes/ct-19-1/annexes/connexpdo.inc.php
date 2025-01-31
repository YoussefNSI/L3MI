<?php
/*
 * construit et renvoie une instance de la classe PDO pour accéder à la base de données
 * de nom $db qui est hébergée sur le serveur local MySQL.
 *
 * @param string $db Nom de la base de données à laquelle se connecter.
 * 
 * @return PDO $pdo L'objet de connexion à la base de données.
 * 
 */
function connexpdo($db)
{
    $sgbd = "mysql"; // choix de MySQL
    $host = "localhost";
    $charset = "UTF8";
    $user = ""; // TODO : user id
    $pass = ""; // TODO : password
    try {
        $pdo = new PDO("$sgbd:host=$host;dbname=$db;charset=$charset", $user, $pass, array(
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION
        ));
        return $pdo;
    } catch (PDOException $e) {
        echo $e->getMessage();
        exit();
    }
}
?>