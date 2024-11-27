<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="../style.css">
    <title>2023 CC-PHP Exercice 3</title>
</head>
<body>
    <form method="post" action="<?= $_SERVER['PHP_SELF'];?>" >
        <label for="fichier">XML à importer:
            <input type="file" name="fichier" id="fichier" accept=".xml">
        </label>
        <input type="submit" name="submit" value="Ajouter à la base de données">
    </form>

    <?php
        // A COMPLETER
    ?>
</body>
</html>