<?php

// A COMPLETER

$html = <<<HTML
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Votre BMI nous importe</title>
</head>
<body>
Symboles :<br/>
<strong>i</strong> : monnaie du pays de résidence<br/>
<strong>j</strong> : monnaie du pays de destination<br/>
<br/>
Calcul :
<ul>
<li>Aller-retour (km) : $allerRetour<br/>
<li>Coût du trajet (à raison de &dollar;0.15/km) : $coutTrajet <strong>i</strong></li>
<li>Prix local du Big Mac dans le pays de résidence : $iLocalPrice <strong>i</strong></li>
<li>Prix local du Big Mac dans le pays de destination : $jLocalPrice <strong>j</strong></li>
<li>Taux de change <sub>i&#47;j</sub> : $TCij</li>
<li>Gain par Big Mac : $gainParBigMac <strong>i</strong></li>
<li>Gain par jour (à raison de 8 Big Macs par jour) : $gainParJour <strong>i</strong></li>
<li>Durée d'amortissement : $dureeAmortissement jours</li>
</ul>

<a href="../exercice-1/bmi.html">Back to Earth</a>
</body>
</html
HTML;

echo $html;
?>
