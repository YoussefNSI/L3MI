<?php
function connexpdo(string $db)
{
    $sgbd = "mysql"; // choix de MySQL (fonctionnera aussi avec MariaDB !)
    $host = "localhost";
    $port = 3306; // port par défaut de MySQL (à adapter selon votre config et votre choix entre mysql/mariadb)
    $charset = "UTF8";

    // A MODIFIER
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

// JS: OUTILS ===============================================


function js(string $code)
{
    echo "<script type=\"text/javascript\">$code</script>";
}

function console(string $str)
{
    js("console.log(\"" . htmlentities($str) . "\");");
}

function alert(string $str)
{
    js("alert(\"$str\");");
}

function displayException(PDOException $e)
{
    console("Fichier : " . $e->getFile());
    console("Ligne : " . $e->getLine());
    console($e->getMessage());
    alert("Code SQL : {$e->getCode()}");
    exit();
}

?>