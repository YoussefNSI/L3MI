<?php
session_start();
$fond=$_SESSION['fond'];
$texte=$_SESSION['texte'];
?>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8" />
    <title>TP PHP - Personnalisation avec sessions</title>
    <style type="text/css">
        body{background-color: <?= "$fond"; ?> ; color: <?= $texte; ?> ;}
        legend {
            font-weight: bold;
            font-family: cursive;
        }

        label {
            font-weight: bold;
            font-style: italic;
        }
    </style>
</head>
<body>
   <p>Contenu de la page B avec les couleurs choisies <br />
   <a href="sessions.php">Retour vers la page principale</a>
</p>
</body>
</html>