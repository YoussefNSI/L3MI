<?php
require "connexpdo.inc.php";

class Correcteur
{

    // nom du fichier de spécification (JSON)
    private $filename;

    // tableau PHP représentant la spécification JSON
    private $form;

    // la superglobale $_GET ou $_POST selon ce qu'indique la spécification
    private $method;

    // tableau des corrections : exemplaire complet fourni dans formulaire-063-correction.txt
    private $correction = [];

    // tableau des points : 1 valeur (dans [0,1]) par champ
    private $points = [];

    // note finale dans [0,1] avec 2 chiffres max après la virgule
    private $note = 0.0;

    // lit le fichier $filename et initialise les propriétés $filename, $form et $method
    public function lire_spec($filename)
    {
        $this->filename = $filename;
        if (! file_exists($filename)) {
            throw new Exception("Le fichier $filename n'existe pas !");
        } else {
            $this->form = json_decode(file_get_contents($filename), true);
        }

        // A COMPLETER : initialiser $method
    }

    // construit $vérification, enregistre la note en BDD, affiche les résultats
    public function vérifier()
    {
        $this->vérifier_champs_textuels();
        $this->vérifier_cases();
        $this->enregistrer_note();
        $this->afficher_résultats();
    }

    // construit $vérification en vérifiant chaque champ texte
    // met à jour $points avec les points (0 ou 1) obtenus pour chaque champ texte
    private function vérifier_champs_textuels()
    {
        // A COMPLETER
    }

    // complète $vérification en vérifiant chaque groupe de cases à cocher
    // met à jour $points avec les points (nombre dans [0,1]) obtenus pour chaque groupe
    private function vérifier_cases()
    {
        // A COMPLETER
    }

    // calcule et enregistre la note en BDD à partir du tableau des points $points
    private function enregistrer_note()
    {
        // A COMPLETER
    }

    // extrait de la BDD toutes les notes obtenues pour le formulaire puis importe le gabarit
    private function afficher_résultats()
    {
        // A COMPLETER
 
        require "gabarit.php";
    }
}

$filename = "formulaire.json";
$c = new Correcteur();
$c->lire_spec($filename);
$c->vérifier();
?>
