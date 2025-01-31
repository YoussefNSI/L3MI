<?php
require("auteurs.php");
require("Livre.php");

$livres = array(); // Tableau contenant l'ensemble des livres (Objet Livre).

// TODO
// Remplir le tableau $livres pour chaque livre trouvé dans le tableau $auteurs (défini dans le fichier auteurs.php)



// Partie affichage
if(__FILE__ == $_SERVER["SCRIPT_FILENAME"]){
echo "<pre>";
print_r($livres);
echo "</pre>";
}
?>
