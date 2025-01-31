<?php
if(isset($_POST['fond']) and isset($_POST['texte'])) { // fails on first execution, succeeds afterwards with fond="" and texte="" if fields are left empty.
    if(!isset($_COOKIE['fond']) AND !isset($_COOKIE['texte']) ) {
        $fond=$_POST['fond'];
        $texte=$_POST['texte'];
        $expir=time() + 10;
        setcookie("fond",$fond,$expir);
        setcookie("texte",$texte,$expir);
    } else {
        $fond=$_COOKIE['fond'];
        $texte=$_COOKIE['texte'];
    }
} else {
    $fond = "white";
    $texte = "black";
}
?>
<!DOCTYPE html>
<html>

<head>
    <meta charset="UTF-8" />
    <title>TP PHP - Personnalisation avec cookies</title>
    <style type="text/css">
        body{background-color: <?=$fond?> ; color: <?=$texte?> ;}
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
