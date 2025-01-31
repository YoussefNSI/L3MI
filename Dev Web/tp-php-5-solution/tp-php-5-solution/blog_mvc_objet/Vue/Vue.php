<?php
require_once 'vendor/autoload.php';

class Vue
{

    // Nom du fichier associé à la vue
    private $fichier;

    public function __construct($action)
    {
        // Détermination du nom du fichier vue à partir de l'action
        $this->fichier = "vue" . $action . ".html";
    }

    // Génère et affiche la vue
    public function generer($donnees)
    {
        $loader = new \Twig\Loader\FilesystemLoader('Vue');
        $twig = new \Twig\Environment($loader); /* no template caching : , array('cache' => '.'));*/
        $template = $twig->load($this->fichier);
        echo $template->render($donnees);
    }
}