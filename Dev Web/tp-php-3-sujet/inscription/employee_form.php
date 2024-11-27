<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8"/>
<title>TP PHP - Inscription d'employés</title>
</head>
<body style="background-color: #ffcc00;">
<form action="<?php echo $_SERVER['PHP_SELF'] ?>" method="post">
<fieldset>
<legend><b>Inscrire un employé</b></legend>
<label>Nom :&nbsp;&nbsp;&nbsp;&nbsp;</label>
<input type="text" name="nom" value="" size="30" maxlength="60" required="required"/><br/><br/>
<label>Salaire :&nbsp;</label>
<input type="number" name="salaire" min="0" max="100000" step="5000" size="6" required="required"/><br/><br/>
<label>Age :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</label>
<input type="number" name="age" min="18" max="100" size="6" required="required"/><br/><br/>
<input type="submit" value="Inscrire" name="inscrire" />
</fieldset>
</form>
<?php
session_start();

if (!isset($_SESSION['employes_array'])) {
    $_SESSION['employes_array'] = array();
}

if (isset($_POST['inscrire'])) {
    $nom = $_POST['nom'];
    $salaire = $_POST['salaire'];
    $age = $_POST['age'];
    $employe = array($nom, $salaire, $age);
    if(in_array($employe, $_SESSION['employes_array'])) {
        echo $nom . " d'âge " . $age . ": Cet employé existe déjà";
    }
    else {
        $_SESSION['employes_array'][] = $employe;
    };
}

$employes = $_SESSION['employes_array'];

if (count($employes) > 0) {
    echo "<table border='1'>";
    echo "<tr><th>Nom</th><th>Salaire</th><th>Age</th></tr>";
    foreach ($employes as $employe) {
        echo "<tr>";
        foreach ($employe as $info) {
            echo "<td>$info</td>";
        }
        echo "</tr>";
    }
    echo "</table>";
}
?>
</body>
</html>
