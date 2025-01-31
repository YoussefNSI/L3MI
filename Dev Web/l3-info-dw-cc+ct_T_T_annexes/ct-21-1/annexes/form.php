<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8"/>
    <title>EDT</title>
    <style type="text/css">
        body {
            background-color: #ffcc00;
        }
        table, th, td {
            border: 1px solid gray;
        }
        table {
            margin: 30px;
            border-collapse: collapse;
        }
        td {
            padding: 5px;
        }
    </style>
</head>
<body>
<form action="<?php $_SERVER['PHP_SELF'] ?>" method="post">
    <fieldset>
    <legend><b>Ajout de groupe</b></legend>
    <label>Partie de cours :&nbsp;</label>
    <select name="part" required>
    <?php
    require_once 'connexion.php';

    // A COMPLETER" 

    ?>
    </select>
    <br/><br/>
    <label>Nom de groupe : </label>
    <input type="text" name="groupName" size="30" required/><br/><br/>
    <label>Effectif maximum :&nbsp;</label>
    <input type="number" name="maxHeadCount" value="10" min="1" max="100" required/><br/><br/>
    <input type="submit" value="Ajouter"/>
    </fieldset>
</form>
<?php


        // INSERTION D'UN NOUVEAU GROUPE

        // AFFICHAGE DE LA PARTIE MODIFIEE

?>
</body>
</html>
