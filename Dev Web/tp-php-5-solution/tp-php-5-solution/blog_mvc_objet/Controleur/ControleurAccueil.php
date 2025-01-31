<?php

require_once 'Modele/Billet.php';
require_once 'Modele/Commentaire.php';
require_once 'Vue/Vue.php';

class ControleurAccueil {

    private $billet;
    private $commentaire;
    
    public function __construct() {
        $this->billet = new Billet();
        $this->commentaire = new Commentaire();
    }

// Affiche la liste de tous les billets du blog
    public function accueil() {
        $billets = $this->billet->getBillets();
        $vue = new Vue("Accueil");
        $vue->generer(array('billets' => $billets));
    }

    // Supprime un billet
    public function supprimer($idBillet) {
        $billet = $this->billet->getBillet($idBillet);
        $this->commentaire->supprimerCommentaires($idBillet);
        $this->billet->supprimerBillet($idBillet);
        
        $this->accueil();
    }
}

