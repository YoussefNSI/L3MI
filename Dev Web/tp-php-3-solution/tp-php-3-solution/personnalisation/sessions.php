<?php
session_start();
$_SESSION['fond'] = (! isset($_POST['fond']) && ! key_exists('fond', $_SESSION)) ? "white" : (empty($_POST['fond']) ? $_SESSION['fond'] : $_POST['fond']);
$_SESSION['texte'] = (! isset($_POST['texte']) && ! key_exists('texte', $_SESSION)) ? "black" : (empty($_POST['texte']) ? $_SESSION['texte'] : $_POST['texte']);

?>
<!DOCTYPE html>
<html>

<head>
    <meta charset="UTF-8" />
    <title>TP PHP - Personnalisation avec sessions</title>
    <style type="text/css">
        body{background-color: <?= $_SESSION['fond'] ?> ; color: <?= $_SESSION['texte'] ?> ;}
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
    <form method="post" action="sessions.php">
        <fieldset>
            <legend>Choisissez vos couleurs (mot clé ou code)</legend>
            <label>Couleur de fond
                <input type="text" name="fond" />
            </label><br /><br />
            <label>Couleur de texte
                <input type="text" name="texte" />
            </label><br />
            <input type="submit" value="Envoyer" />&nbsp;&nbsp;
            <input type="reset" value="Effacer" />
        </fieldset>
    </form>

    <p>Contenu de la page principale <br />
        <a href="sessions-B.php">Lien vers la page B qui aura ces couleurs</a>
    </p>
</body>
</html>
