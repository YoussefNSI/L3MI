<?php

require_once 'Framework/Controleur.php';
require_once 'Modele/Billet.php';
require_once 'Modele/Commentaire.php'; // AJOUT

class ControleurAccueil extends Controleur {

    private $billet;
    private $commentaire;
    
    public function __construct() {
        $this->billet = new Billet();
        $this->commentaire = new Commentaire();
    }

    // Affiche la liste de tous les billets du blog
    public function index() {
        $billets = $this->billet->getBillets();
        $this->genererVue(array('billets' => $billets));
    }

    // AJOUT
    // Supprime un billet
    public function supprimer() {
        $idBillet = $this->requete->getParametre("id");
        $this->commentaire->supprimerCommentaires($idBillet);
        $this->billet->supprimerBillet($idBillet);
        
        $this->executerAction("index");
    }
}