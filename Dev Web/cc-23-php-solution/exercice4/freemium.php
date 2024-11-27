<?php 
    /*
        Freemium :  Mot-valise des mots anglais 'free' (gratuit) et 'premium' (prime).

        Le modèle FREEMIUM s'agit souvent d'une version limitée dans le temps 
        servant à promouvoir une version complète payante.
     */


    $allow_map = false;
    if (isset($_POST['submit']) && isset($_POST['map'])) 
    {
        $map_name = $_POST['map'];
        if (isset($_COOKIE[$map_name])) {
            if ($_COOKIE[$map_name] < 3) {
                // allow map visualization only if cookie's value is < 3
                $allow_map = true;
                // reset the cookie with the new value
                setcookie($map_name, $_COOKIE[$map_name]+1, (time()+60*24*30*12)); 
            }
        }
        else {
            // setup a cookie (for a very long time)
            // echo "setting up a cookie.. ";
            $allow_map = true;
            setcookie($map_name, 1, (time()+60*24*30*12));
        }
    }
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;1,300;1,400;1,500&display=swap" rel="stylesheet">
    <style type="text/css">
        * { font-family: 'Roboto', sans-serif; text-align: center;}
        .error { color: red }
        select, input { padding: 3px; }
    </style>
    <title>2023 CC-PHP Exercice 4</title>
</head>
<body>
    <h2>Service d'affichage de cartes cognitives</h2>
    <form method="post" action="<?= $_SERVER['PHP_SELF'];?>" >
        <label for="map">Choissez une carte cognitive :
            <select name="map" id="map">
                <option value="map1">Map_1</option>
                <option value="map2">Map_2</option>
                <option value="map3">Map_3</option>
            </select>
        </label>
        <input type="submit" name="submit" value="Afficher">
    </form>

    <?php
        if (isset($_POST['map']) && ($allow_map == false)) {
            echo "<p class='error'>Vous ne pouvez plus afficher cette carte : nombre maximum d'affichages atteint).</p>";
        }
    ?>
</body>
</html>