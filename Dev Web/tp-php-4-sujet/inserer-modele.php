<!DOCTYPE html >
<html>
<head>
<meta charset="UTF-8" />
<title>Saisissez les caractéristiques du modèle</title>
</head>
<body>
	<form action="<?php echo $_SERVER['PHP_SELF'];?>" method="post"
		enctype="application/x-www-form-urlencoded">
		<fieldset>
			<legend>
				<b>Modèle de voiture</b>
			</legend>
			<table>
				<tr>
					<td>Code :</td>
					<td><input type="text" name="id_modele" size="40" maxlength="30"/></td>
				</tr>
				<tr>
					<td>Nom du modèle :</td>
					<td><input type="text" name="modele" size="40" maxlength="30"/></td>
				</tr>
				<tr>
					<td>Carburant : <select name="carburant">
							<option value="essence">Essence</option>
							<option value="diesel">Diesel</option>
							<option value="gpl">G.P.L.</option>
							<option value="électrique">Electrique</option>
					</select></td>
				</tr>
				<tr>
					<td><input type="reset" value=" Effacer "></td>
					<td><input type="submit" value=" Envoyer " name="submit"></td>
				</tr>
			</table>
		</fieldset>
	</form>
<?php
if(isset($_POST["submit"])) {
	if(!empty($_POST["id_modele"]) && !empty($_POST["modele"]) && !empty($_POST["carburant"])){
		try {
			require ("connexpdo.inc.php");
			$objdb = connexpdo("voitures");
			$str = "INSERT INTO modele (id_modele, modele, carburant) VALUES (:id_modele, :modele, :carburant)";
			$stmt = $pdo->prepare($str);
			$stmt->execute([
				'id_modele' => $_POST['id_modele'],
				'modele' => $_POST['modele'],
				'carburant' => $_POST['carburant']
			]);
			echo "Voiture enregistrée dans la base de données";
		} catch (PDOException $e) {
			echo "Erreur: " . $e->getMessage();
		}
	}
}
	
?>
</body>
</html>