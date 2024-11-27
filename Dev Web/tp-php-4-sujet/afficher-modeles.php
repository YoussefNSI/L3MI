<!DOCTYPE html >
<html>
<head>
<meta charset="UTF-8" />
<title>Lecture de la table modele</title>
<style type="text/css">
table, tr, td, th {
	border-style: solid;
	border-color: red;
	background-color: yellow;
}
table {
	border-width: 3px;
	border-collapse: collapse;
}
tr, td, th {
	border-width: 1px;
}
</style>
</head>
<body>
<?php
require ("connexpdo.inc.php");
$objdb = connexpdo("voitures");
$qry = " SELECT * FROM modele ORDER BY modele";
$stt = $objdb->query($qry);
echo "<table>";
echo "<tr>
		<th> Code modèle</th>
		<th> Modèle</th>
		<th> Carburant</th>
	</tr>";
while ($record=$stt->fetch(PDO::FETCH_BOTH)) {
	echo "<tr>";
	echo "<td>" . $record["id_modele"] . "</td>";
	echo "<td>" . $record["modele"] . "</td>";
	echo "<td>" . $record["carburant"] . "</td>";
	echo "</tr>";
}
echo "</table>";
?>
</body>
</html>