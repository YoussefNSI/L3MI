<?php
// Calcul et génération taxe et coût TTC par article sous forme de ligne de tableau HTML
// $value : valeur de type array d'un élément du tableau $prix_taux
// $key : clé de type string d'un élément du tableau $prix_taux
// $param : paramètre additionnel de type string (couleur de fond CSS)
//

function taxe($value, $key, $param)
{
    $prix = $value["Prix"];
    $taux = $value["Taux"];
    $taxe = $prix * $taux;
    $prix_ttc = $prix + $taxe;
    return "<tr style='background-color:$param'><td>$key</td><td>$prix</td><td>$taux</td><td>$taxe</td><td>$prix_ttc</td></tr>";
}

// Génération de tableau HTML
//
function generer_tableau($prix_taux)
{
    echo <<<HTML
<table border='1'>
    <tr>
        <th>Article</th>
        <th>Prix</th>
        <th>Taux T.V.A.</th>
        <th>Taxe</th>
        <th>Coût T.T.C.</th>
    </tr>
HTML;
    $couleur = "white";
    array_walk($prix_taux, function($value, $key) use (&$couleur) {
        echo taxe($value, $key, $couleur);
        $couleur = $couleur == "white" ? "lightgrey" : "white";
    });
    echo "</table>";
}

// tri du tableau

usort($prix_taux, function($a, $b) {
    return $a["Taux"] <=> $b["Taux"];
});

// Affichage du tableau
generer_tableau($prix_taux);
?>