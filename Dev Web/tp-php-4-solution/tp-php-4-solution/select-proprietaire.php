<?php
include_once ("connexpdo.inc.php");
try {
    $idcom = connexpdo('l3_dw_tp_php_bdd_voitures');
    $qry = "SELECT `id_pers`, `nom`, `prenom` FROM `proprietaire`";
    $result = $idcom->query($qry);
    echo "<tr><td>Nom & prénom</td><td><select name=\"id_pers\">\n";
    while ($row = $result->fetch(PDO::FETCH_ASSOC)) {
        $id = $row['id_pers'];
        $nom = $row['nom'];
        $prenom = $row['prenom'];
        $selected = (isset($_POST['id_pers']) && $_POST['id_pers']==$id)  ? "selected=\"selected\"" : "";
        echo "<option value=\"$id\" $selected>$nom $prenom</option>\n";
    }
    echo "</select></td></tr>\n";
    $result->closeCursor();
} catch (PDOException $e) {
    displayException($e);
    exit();
}
?>