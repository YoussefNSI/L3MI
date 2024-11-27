<!DOCTYPE html>
<html>
<head>
<title>Téléversement de fichier</title>
</head>
<body>
	<form action="<?= $_SERVER['PHP_SELF'] ?>" method="post"
		enctype="multipart/form-data">
		<fieldset>
			<legend>
				<b>Transférez un fichier ZIP</b>
			</legend>
			<table border="1">
				<tr>
					<td>Choisissez un fichier</td>
					<td><input type="file" name="fich" accept="application/zip" /></td>
					<td><input type="hidden" name="MAX_FILE_SIZE" value="1000000" /></td>
				</tr>
				<tr>
					<td>&nbsp;</td>
					<td><input type="submit" value="ENVOI" /></td>
				</tr>
			</table>
		</fieldset>
	</form>
</body>
</html>
<?php
if ($_FILES) {
	$fich = $_FILES['fich'];
	if ($fich['error'] == 0) {
		$nom = $fich['name'];
		$taille = $fich['size'];
		$type = $fich['type'];
		if($taille > 1000000) {
			echo "Fichier trop volumineux";
			exit;
		}
		if($type != "application/zip" && $type != "application/x-zip-compressed") {
			echo "Type de fichier incorrect";
			exit;
		}
		echo "<h2>Vous avez bien transféré le fichier </h2>";
		echo "<p>Nom du fichier : $nom</p>";
		echo "<p>Taille du fichier : $taille octets</p>";
	} else {
		echo "Erreur de transfert";
	}
}
?>