<!DOCTYPE html >
<html>
<head>
<meta charset="UTF-8" />
<title>Saisissez les caractéristiques du modèle</title>
</head>
<body>
	<form action="<?php echo $_SERVER['PHP_SELF'];?>" name="form1"
		method="post" enctype="application/x-www-form-urlencoded">
		<fieldset>
			<legend>
				<b>Enregistrement d'un véhicule</b>
			</legend>
			<table>
				<tr>
					<td colspan="2"><b>Propriétaire</b></td>
				</tr>
<?php require_once ("select-proprietaire.php");?>
				<tr>
					<td colspan="2"><b>Voiture</b></td>
				</tr>
<?php require_once ("select-voiture.php");?>
				<tr>
					<td colspan="2"><b>Carte grise</b></td>
				</tr>
<?php require_once ("input-cartegrise.php");?>
				<tr>
					<td><input type="reset" value="Effacer" /></td>
					<td><input type="submit" name="enreg" value="ENREGISTRER"/></td>
				</tr>
			</table>
		</fieldset>
	</form>
<?php
include_once ("connexpdo.inc.php");
$idcom = connexpdo('l3_dw_tp_php_bdd_voitures');
if (isset($_POST['enreg'])) {
    $id_pers = $idcom->quote($_POST['id_pers']);
    $voiture = $idcom->quote($_POST['voiture']);
    $datecarte = $idcom->quote($_POST['datecarte']);
    try {
        $query = "INSERT INTO `cartegrise` VALUES ($id_pers,$voiture,$datecarte)";
        $result = $idcom->query($query);
        alert('La carte grise est enregistrée !');
    } catch (PDOException $e) {
        displayException($e);
        exit();
    }
} else {
    echo "<h3>Formulaire à compléter!</h3>";
}
?>
 </body>
</html>