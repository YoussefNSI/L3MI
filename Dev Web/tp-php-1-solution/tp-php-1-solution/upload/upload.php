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
if (isset($_FILES['fich'])) {
    if ($_POST["MAX_FILE_SIZE"] < $_FILES["fich"]["size"]) {
        echo "<b>Taille trop grande </b><hr />";
        echo "Taille maximale autorisée :", $_POST["MAX_FILE_SIZE"], " octets<hr / >";
        echo "Taille du fichier transféré :", $_FILES["fich"]["size"], " octets<hr / >";
    } else {
        // Enregistrement et renommage du fichier
        $result = move_uploaded_file($_FILES["fich"]["tmp_name"], "monfichier.zip");
        if ($result == TRUE) {
            echo "<b>Vous avez bien transféré le fichier</b><hr />";
            echo "Le nom du fichier est : ", $_FILES["fich"]["name"], "<hr/>";
            echo "Votre fichier a une taille de ", $_FILES["fich"]["size"], "<hr />";
        } else {
            echo "<hr /> Erreur de transfert n°", $_FILES["fich"]["error"];
        }
    }
}
?>