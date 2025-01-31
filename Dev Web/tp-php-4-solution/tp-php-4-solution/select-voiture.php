<tr>
	<td>Immatriculation :</td>
	<td>
<?php
$immat = $_POST['immat'] ?? "";
$value = "value=\"$immat\"";
echo "<input type=\"text\" name=\"immat\" $value size=\"20\" maxlength=\"20\"/>";
?>
	<input type="submit" name="submit-immat" value="Chercher les voitures"/>
	</td>
</tr>
<?php
include_once ("connexpdo.inc.php");
if (isset($_POST['submit-immat']) && isset($_POST['immat'])) {
    try {
        $idcom = connexpdo('l3_dw_tp_php_bdd_voitures');
        $immat = $_POST['immat'];
        $query = "SELECT DISTINCT `immat` AS `id_immat` FROM `voiture` WHERE `immat` LIKE '%$immat%'";
        $result = $idcom->query($query);
        echo "<tr><td>Les voitures</td><td><select name=\"voiture\">\n";
        while ($row = $result->fetch(PDO::FETCH_ASSOC)) {
            $id = $row['id_immat'];
            $checked = (isset($_POST['voiture']) && $_POST['voiture'] == $id) ? "checked=\"checked\"" : "";
            echo "<option value=\"$id\" $checked>$id</option>\n";
        }
        echo "</select></td></tr>\n";
        $result->closeCursor();
    } catch (PDOException $e) {
        displayException($e);
        exit();
    }
}
?>