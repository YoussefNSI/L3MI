<?php

require_once 'Configuration.php';
require_once './vendor/autoload.php'; // AJOUT POUR TWIG

/**
 * Classe modélisant une vue
 *
 * @version 1.0
 * @author Baptiste Pesquet
 */
class Vue {

    /** Nom du fichier associé à la vue */
    private $fichier;

    /** Titre de la vue */
    private $titre;

    /**
     * Constructeur
     * 
     * @param string $action Action à laquelle la vue est associée
     * @param string $controleur Nom du contrôleur auquel la vue est associée
     */
    public function __construct($action, $controleur = "") {
        // Détermination du nom du fichier vue à partir de l'action et du constructeur
        // La convention de nommage des fichiers vues est : Vue/<$controleur>/<$action>.php
        $fichier = "";
        if ($controleur != "") {
            $fichier = $fichier . $controleur . "/";
        }
        $this->fichier = $fichier . $action . ".html";
    }

    // MODIFICATION POUR USAGE DE TWIG
    /**
     * Génère et affiche la vue
     * 
     * @param array $donnees Données nécessaires à la génération de la vue
     */
    public function generer($donnees) {
        $loader = new \Twig\Loader\FilesystemLoader('Vue');
        $twig = new \Twig\Environment($loader); /* no template caching : , array('cache' => '.'));*/
        $template = $twig->load($this->fichier);
        $racineWeb = Configuration::get("racineWeb", "/");
        $donnees['titre'] = $this->titre;
        $donnees['racineWeb'] = $racineWeb;
        echo $template->render($donnees);
    }
}
