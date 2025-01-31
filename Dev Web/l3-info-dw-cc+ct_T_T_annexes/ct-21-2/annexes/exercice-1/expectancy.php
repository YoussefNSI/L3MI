<?php
require("connexpdo.inc.php"); // pensez à modifier la fonction connexpdo pour vous connecter à la base de données.
// TODO : Q1) Récupérez en base de données les pays (id, nom, continent et espérance de vie) : le résultat sera stocké dans $pays
$pays = array();

?>
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Pays et espérance de vie</title>
    <style>
        .msg {
            padding: 10px;
            width: 50%;
            text-align: center;
            background-color: powderblue;
        }
        .error {
            padding: 10px;
            width: 50%;
            text-align: center;
            background-color: lightcoral;
        }
    </style>
</head>
<body>
<form action="" method="POST">
        <fieldset>
            <legend>Modifier espérance de vie d'un pays</legend>
            <table style="text-align:right">
                <tr>
                    <td>Pays :</td>
                    <td>
                        <select name="pays" id="pays">
                            <?php
                            // TODO : Q2) Générez les options du select contenant tous les pays de la $pays (value=ID et le texte est le nom du pays). Si l'id du pays est envoyé par le formulaire l'attribut selected est ajouté à l'option du pays correspondant à l'ID.
                            // Si vous n'avez pas réussi la question d'avant décommentez la ligne suivante qui importe les éléments du tableau $pays :
                            // require("data/pays.php");
                            
                            ?>
                        </select>
                    </td>
                    <td><input type="submit" name="remplir" value="Remplir"></td>
                </tr>
                <tr>
                    <td>Nom :</td>
                    <?php
                    // TODO : Q3) Si l'utilisateur a cliqué sur le bouton "Remplir" renseigner la valeur de $expectancy sinon elle sera égale à la moyenne des espérances de vie trouvées dans le tableau $pays (Attention aux valeurs NULL qui ne seront pas considérées)
                    // Si vous n'avez pas réussi la Q1 décommentez la ligne suivante qui importe les éléments du tableau $pays :
                    // require("data/pays.php");
                    // Si vous n'avez pas réussi la Q2 décommentez la ligne suivante qui importe les options de la balise select :
                    // echo "<script src=\"data/select.js\"></script>";
                    $expectancy = 66.40;
                    
                    ?>
                    <td><input type="text" name="expectancy" id="expectancy" value="<?= $expectancy ?>" ></td>
                </tr>
                <tr>
                    <td></td><td><input type="submit" name="modifier" value="Modifier"></td>
                </tr>
            </table>
        </fieldset>
    </form>
    <?php
    // TODO : Q4) Si l'utilisateur a cliqué sur le bouton "Modifier", modifiez la valeur de l'espérance de vie du pays sélectionné dans la base de données. Avant vous vérifierez que la valeur de l'espérance de vie est comprise entre 0 et 100 ans.
    // Si vous n'avez pas réussi la Q2 décommentez la ligne suivante qui importe les options de la balise select :
    // echo "<script src=\"data/select.js\"></script>";
    
    ?>
    <div>
        <h2>Espérance de vie moyenne par continent :</h2>
        <?php
        // TODO : Q5) A partir de $pays remplissez le tableau clef-valeur $expectCont contenant l'espérance de vie moyenne par continent (où le continent est la clef et la valeur est l'espérance de vie moyenne);
        // Si vous n'avez pas réussi la Q1 décommentez la ligne suivante qui importe les éléments du tableau $pays :
        // require("data/pays.php");
        $expectCont = array();
        

        // TODO : Q6) Affichez une liste à puces contenant l'ensemble des continents et de leur espérance de vie moyenne (tri par ordre décroissant de l'espérance de vie)
        // Si vous n'avez pas réussi la Q5 décommentez la ligne suivante qui importe les éléments du tableau $expectCont :
        // require("data/expectCont.php");
        
        ?>
    </div>
</body>
</html>
