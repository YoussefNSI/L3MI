<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<title>CT JS/PHP : Partie 1 - PHP</title>
<style type="text/css">
.error { 
    color : red;
    background-color : rgba(255, 0, 0, .3);
}
.success { 
    color : green;
    background-color : rgba(0, 255, 0, .3);
}
div{
    margin-top: 10px;
}
</style>
</head>
<body>
<?php
// CONNEXION A LA BASE
include ("exercice1.inc.php");
$idcom = connexpdo("l3info_ct_20_2_dbsport");
?>
<h1>Ajouter une équipe</h1>
<div>
<form action="<?php echo $_SERVER['PHP_SELF'];?>" name="formEquipe"
	method="post">
<label><b>Nom du club : </b></label><input type="text" name="nom" maxlength="15" required><br /><br />
<label><b>Couleur du maillot : </b></label><input type="text" name="cmaillot" maxlength="30" required><br /><br />
<label><b>Prestige : </b></label><input type="number" name="prestige" value="0" max="127" min="-128" required><br /><br />
<input type="submit" name="addEquipe" value="Ajouter">
</form>
</div>
<?php

    // A COMPLETER
    // n'hésitez pas à utiliser la variable $idcom définie plus haut
    

?>
</body>
</head>
</html>
