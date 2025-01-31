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
include ("connexpdo.inc.php");
$idcom = connexpdo("l3_dw_tp_php_bdd_voitures");
$query = "SELECT id_modele AS 'Code modèle', modele AS 'Modèle', carburant AS 'Carburant' FROM modele ORDER BY modele";
try {
    $str = "<table>";
    $result = $idcom->query($query);
    $r = 1;
    while ($row = $result->fetch(PDO::FETCH_ASSOC)) {
        if ($r == 1) {
            $str .= "<tr>";
            foreach ($row as $key => $val) {
                $str .= "<th>$key</th>";
            }
            $str .= "</tr>";
            $r ++;
        }
        $str .= "<tr>";
        foreach ($row as $val) {
            $str .= "<td>$val</td>";
        }
        $str .= "</tr>";
    }
    echo $str . "</table>";
} catch (PDOException $e) {
    displayException($e);
    exit();
}
$idcom = null;
?>
</body>
</html>