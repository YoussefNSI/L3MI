<?php 
    /*
        Freemium :  Mot-valise des mots anglais 'free' (gratuit) et 'premium' (prime).

        Le modèle FREEMIUM s'agit souvent d'une version limitée dans le temps 
        servant à promouvoir une version complète payante.
     */


    $allow_map = false;
    if(isset($_POST['SUBMIT'])){
        $map_name = $_POST['map'];
        if(isset($_COOKIE[$map_name])){
            if($_COOKIE[$map_name] < 3){
                $allow_map = true;
                setcookie($map_name, $_COOKIE[$map_name]+1, time()+3600*24*30);
            }
        }else{
            setcookie($map_name, 1, time()+3600*24*30);
            $allow_map = true;
        }
    }
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="../style.css">
    <title>2023 CC-PHP Exercice 4</title>
</head>
<body>
    <h2>Service d'affichage de cartes cognitives</h2>
    <form method="post" action="<?= $_SERVER['PHP_SELF'];?>" >
        <label for="map">Choissez une carte cognitive:
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
            echo "<p class='error'>Vous avez dépassé le nombre de visualisation autorisé pour cette carte cognitive</p>";
        }
    ?>
</body>
</html>