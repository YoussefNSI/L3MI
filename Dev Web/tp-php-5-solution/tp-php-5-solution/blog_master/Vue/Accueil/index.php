<?php $this->titre = "Mon Blog"; ?>

<?php foreach ($billets as $billet):
    ?>
    <article>
        <header>
            <a href="<?= "billet/index/" . $this->nettoyer($billet['id']) ?>">
                <h1 class="titreBillet"><?= $this->nettoyer($billet['titre']) ?></h1>
            </a>
            <form method="post" action="accueil/supprimer">
            <input type="hidden" name="id" value="<?= $billet['id'] ?>" />
            <input type="submit" value="Supprimer" /></form> 
            <time><?= $this->nettoyer($billet['date']) ?></time>
        </header>
        <p><?= $this->nettoyer($billet['contenu']) ?></p>
    </article>
    <hr />
<?php endforeach; ?>
