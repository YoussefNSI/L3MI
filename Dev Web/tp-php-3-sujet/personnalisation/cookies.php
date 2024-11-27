<!DOCTYPE html>
<html>

<head>
    <meta charset="UTF-8" />
    <title>TP PHP - Personnalisation avec cookies</title>
    <style type="text/css">
        <?php
        if (isset($_POST['fond'])) {
            echo 'body {background-color: ' . $_POST['fond'] . ';}';
            setcookie('fond', $_POST['fond'], time() + 10);
        }
        if (isset($_POST['texte'])) {
            echo 'body {color: ' . $_POST['texte'] . ';}';
            setcookie('texte', $_POST['texte'], time() + 10);
        }
        if(isset($_COOKIE['fond'])) {
            echo 'body {background-color: ' . $_COOKIE['fond'] . ';}';
        }
        if(isset($_COOKIE['texte'])) {
            echo 'body {color: ' . $_COOKIE['texte'] . ';}';
        }

        ?>
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
    <form method="post" action="cookies.php">
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
</body>

</html>