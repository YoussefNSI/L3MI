<?php

// Connexion à la BdD
function connexpdo(string $db)
{
    $sgbd = "mysql"; // choix de MySQL
    $host = "localhost";
    $charset = "UTF8";
    $user = ""; // user id
    $pass = ""; // password
    try {
        $pdo = new PDO("$sgbd:host=$host;dbname=$db;charset=$charset", $user, $pass);
        // force le lancement d'exception en cas d'erreurs d'exécution de requêtes SQL
        // via eg. $pdo->query()
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        return $pdo;
    } catch (PDOException $e) {
        displayException($e);
        exit();
    }
}

// Génération du select pour les championnat
// Récupération des championnats dans la base
// Création de la variable SESSION "championnats".
function selectChampionnats()
{
    echo "<label><b>Championnat : </b></label><select name=\"id_champ\">\n";
    echo "</select>\n";
}

// Récupération des Equipes et Joueurs dans la base
// Intégration des équipes et joueurs dans la variable SESSION "championnats"
function getEquipesandJoueurs()
{
    
}

// JS useful functions
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
