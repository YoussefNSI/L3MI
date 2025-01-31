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
				<tr>
					<td>Nom :</td>
					<td><input type="text" name="nom"
						<?php if(!empty($_POST['nom'])) echo "value=\"{$_POST['nom']}\""; else echo "value=\"Lauda\""; ?>
						size="40" maxlength="30" /></td>
				</tr>
				<tr>
					<td>Prénom :</td>
					<td><input type="text" name="prenom"
						<?php if(!empty($_POST['prenom'])) echo "value=\"{$_POST['prenom']}\""; else echo "value=\"Niki\""; ?>
						size="40" maxlength="30" /></td>
				</tr>
				<tr>
					<td>Adresse :</td>
					<td><input type="text" name="adresse"
						<?php if(!empty($_POST['adresse'])) echo "value=\"{$_POST['adresse']}\""; else echo "value=\"Route des Hunaudières\""; ?>
						size="40" maxlength="50" /></td>
				</tr>
				<tr>
					<td>Ville :</td>
					<td><input type="text" name="ville"
						<?php if(!empty($_POST['ville'])) echo "value=\"{$_POST['ville']}\""; else echo "value=\"Monte Carlo\""; ?>
						size="40" maxlength="40" /></td>
				</tr>
				<tr>
					<td>Code postal :</td>
					<td><input type="number" name="codepostal" min="01000" max="99999"
						<?php if(!empty($_POST['codepostal'])) echo "value=\"{$_POST['codepostal']}\""; else echo "value=\"98000\""; ?>
						size="5" maxlength="5" /></td>
				</tr>
				<tr>
					<td colspan="2"><b>Modèle</b></td>
				</tr>
				<tr>
					<td>Marque :</td>
					<td><input type="text" name="modele" size="40" maxlength="30" /> <input
						type="submit" name="marque" value="Chercher les modèles" /></td>
				</tr>
<?php
if (isset($_POST['marque'])) {
    include_once ("connexpdo.inc.php");
    $idcom = connexpdo('l3_dw_tp_php_bdd_voitures');
    $modele = $_POST['modele'];
    $requete = "SELECT DISTINCT `id_modele`, `modele` FROM `modele` WHERE `modele` LIKE '%$modele%'";
    $result = $idcom->query($requete);
    echo "<tr><td>Les modèles</td><td><select name=\"id_modele\">";
    while ($ligne = $result->fetch(PDO::FETCH_BOTH)) {
        echo " <option value=\"{$ligne[0]}\">{$ligne[1]}</option>";
    }
    echo "</select></td></tr>";
}
?>
      <tr>
					<td>Carburant :</td>
					<td><select name="carburant">
							<option value="essence">Essence</option>
							<option value="diesel">Diesel</option>
							<option value="électrique">Electrique</option>
							<option value="gpl">G.P.L.</option>
					</select></td>
				</tr>
				<tr>
					<td colspan="2"><b>Voiture</b></td>
				</tr>
				<tr>
					<td>Numéro d'immatriculation</td>
					<td><input type="text" name="immat" value="AB12CD" maxlength="6" /></td>
				</tr>
				<tr>
					<td>Couleur :</td>
					<td><select name="couleur">
							<option value="claire">Claire</option>
							<option value="moyenne">Moyenne</option>
							<option value="foncée">Foncée</option>
					</select></td>
				</tr>
				<tr>
					<td>Date 1ere immatriculation AAAA-MM-JJ</td>
					<td><input type="date" name="datevoiture" value="2017-12-31" /></td>
				</tr>
				<tr>
					<td>Date de la carte grise AAAA-MM-JJ</td>
					<td><input type="date" name="datecarte" value="2018-02-01" /></td>
				</tr>
				<tr>
					<td><input type="reset" value=" Effacer " /></td>
					<td><input type="submit" value="ENREGISTRER" name="enreg" /></td>
				</tr>
			</table>
		</fieldset>
	</form>
<?php
include_once ("connexpdo.inc.php");
$idcom = connexpdo('l3_dw_tp_php_bdd_voitures');
if (isset($_POST['enreg'])) {
    // Récupération des valeurs du formulaire
    $id_modele = $idcom->quote($_POST['id_modele']);
    $carburant = $idcom->quote($_POST['carburant']);
    $immat = $idcom->quote($_POST['immat']);
    $couleur = $idcom->quote($_POST['couleur']);
    $datevoiture = $idcom->quote($_POST['datevoiture']);
    $datecarte = $idcom->quote($_POST['datecarte']);
    $nom = $idcom->quote($_POST['nom']);
    $prenom = $idcom->quote($_POST['prenom']);
    $adresse = $idcom->quote($_POST['adresse']);
    $ville = $idcom->quote($_POST['ville']);
    $codepostal = $idcom->quote($_POST['codepostal']);
    $idcom->beginTransaction();
    $requete = "INSERT INTO `voiture` VALUES($immat,$id_modele,$couleur,$datevoiture)";
    // compteur des requêtes réussies
    $valid = $idcom->exec($requete);
    $requete = "INSERT INTO proprietaire (nom,prenom,adresse,ville,codepostal) VALUES($nom,$prenom,$adresse,$ville,$codepostal)";
    $valid += $idcom->exec($requete);
    $clue = "proprietaire.nom=$nom AND proprietaire.prenom=$prenom AND proprietaire.adresse=$adresse AND proprietaire.ville=$ville AND proprietaire.codepostal=$codepostal";
    $stt = $idcom->query("SELECT `id_pers` FROM `proprietaire` WHERE $clue");
    $id_pers = $stt->fetch(PDO::FETCH_NUM)[0];
    $requete = "INSERT INTO `cartegrise` VALUES($id_pers,$immat,$datecarte)";
    $valid += $idcom->exec($requete);
    if ($valid===3) {
        $idcom->commit();
        echo "<script type=\"text/javascript\">alert('La carte grise est enregistrée ');</script>";
    } else {
        $idcom->rollBack();
        echo "<script type=\"text/javascript\"> console.log(\"".$idcom->errorInfo()[2]."\");</script>";
        echo "<script type=\"text/javascript\"> alert('Erreur d insertion : " . $idcom->errorCode()."');</script>";
    }
} else {
    echo "<h3>Formulaire à compléter!</h3>";
}
?>
 </body>
</html>
