<!DOCTYPE html >
<html>
<head>
<meta charset="UTF-8" />
<title>Recherche des propriétaires d'un modèle</title>
</head>
<body>
	<form action="<?php echo $_SERVER['PHP_SELF'];?>" name="form1"
		method="post" enctype="application/x-www-form-urlencoded">
		<fieldset>
			<legend>
				<b>Choisir le modèle</b>
			</legend>
			<table>
				<tr>
					<td>Marque et modèle</td>
					<td><input type="text" name="modele" maxlength=" " /></td>
				</tr>
				<tr>
					<td><input type="submit" value="Chercher" /></td>
				</tr>
			</table>
		</fieldset>
	</form>
<?php
if (isset($_POST['modele'])) {
    include_once ('connexpdo.inc.php');
    $idcom = connexpdo('l3_dw_tp_php_bdd_voitures');
    $modele = $_POST['modele'];
    $requete = "SELECT proprietaire.nom,proprietaire.prenom FROM voiture,modele,proprietaire,cartegrise WHERE modele LIKE '%$modele%' AND voiture.id_modele=modele.id_modele AND cartegrise.immat=voiture.immat AND proprietaire.id_pers=cartegrise.id_pers";
    $result = $idcom->query($requete);
    echo "<table border=\"1\" >";
    while ($ligne = $result->fetch(PDO::FETCH_BOTH)) {
        echo " <tr><td>$ligne[0]</td><td> $ligne[1]</td>";
    }
    echo "</table>";
} else {
    echo "<h3>Formulaire à compléter!</h3>";
}
?>
</body>
</html>