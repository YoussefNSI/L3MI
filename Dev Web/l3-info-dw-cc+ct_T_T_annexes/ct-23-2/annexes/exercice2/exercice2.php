<!DOCTYPE html>
<html lang="fr">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="../utils/style.css">
    <style>
        #datetime-bar {
            background-color: blue;
            color: white;
        }
    </style>
    <title>CT 2023 JS/PHP</title>
</head>

<body>
    <header id="datetime-bar">
        <p>
            <?php
            echo date('l j F Y') . '<br>';
            echo date('H:i');
            ?>
        </p>
    </header>

    <section>
        <h2>Service d'affichage de cartes cognitives</h2>
        <form method="post" action="<?= $_SERVER['PHP_SELF']; ?>">
            <label for="map">Choisissez une carte cognitive:
                <br>
                <select name="map" id="map">
                    <option value="1">Map_1</option>
                    <option value="2">Map_2</option>
                </select>
            </label>
            <input type="submit" name="submit" value="Afficher">
        </form>
    </section>

    <section id="result">
        <?php
        // A COMPLETER

        ?>
    </section>
    <hr>
    <footer>

    </footer>
</body>

</html>