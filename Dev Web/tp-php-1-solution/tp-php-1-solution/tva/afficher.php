<?php
// Calcul et génération taxe et coût TTC par article sous forme de ligne de tableau HTML
// $value : valeur de type array d'un élément du tableau $prix_taux
// $key : clé de type string d'un élément du tableau $prix_taux
// $param : paramètre additionnel de type string (couleur de fond CSS)
//
// function taxe(array $value,string $key, string $param) : void {
function taxe($value, $key, $param)
{
    $prix = $value["Prix"];
    $taux = $value["Taux"];
    $HT = round($prix * $taux, 2);
    $TTC = round($prix * (1 + $taux), 2);

    $ligne =<<<LIGNE
    <tr>
        <td>$key</td>
        <td class="ra">$prix</td>
        <td class="ra">$taux</td>
        <td class="ra">$HT</td>
        <td class="ra" style="background-color:$param">$TTC</td>
    </tr>
LIGNE;
    echo $ligne;
}

// Génération de tableau HTML
//
// function generer_tableau(array $prix_taux) : void {
function generer_tableau($prix_taux)
{
    $h2table =<<<H2TABLE
    <h2>Facture détaillée en &euro;</h2>
    <table>
        <thead>
            <th>Article</th>
            <th class="ra">Prix</th>
            <th class="ra"><a href="tableau.php">T.V.A.</a></th>
            <th class="ra">Taxe</th>
            <th class="ra">Coût TTC</th>
        </thead>
        <tbody>
H2TABLE;
    echo $h2table;

    array_walk($prix_taux, "taxe", "red");
    $table =<<<TABLE
            </tbody></table>
TABLE;
    echo $table;
}

// tri du tableau
function mon_critère($a, $b) {
	if($a["Taux"] != $b["Taux"]) {
		return (int) 100*($a["Taux"] - $b["Taux"]);
	 } else {
		return (int) 100*($b["Prix"] - $a["Prix"]);
	 }
}
uasort($prix_taux, "mon_critère");

// Affichage tableau
generer_tableau($prix_taux);
?>