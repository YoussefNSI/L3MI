<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="../style.css">
    <title>2023 CC-PHP Exercice 3</title>
</head>
<body>
    <form method="post" action="<?= $_SERVER['PHP_SELF'];?>" >
        <label for="fichier">XML à importer:
            <input type="file" name="fichier" id="fichier" accept=".xml">
        </label>
        <input type="submit" name="submit" value="Ajouter à la base de données">
    </form>

    <?php
    
        include 'xml.php';
        include 'bdd_insert.php';

        if(isset($_FILES["fichier"])){
            $filename = "fichier.xml";
            $uploadfile = __DIR__ . "/$filename";
            if (move_uploaded_file($_FILES['fichier']['tmp_name'], $uploadfile)) {
                echo "Le fichier est valide, et a été téléchargé avec succès.<br>";
            } else {
                echo "Attaque potentielle par téléchargement de fichiers.<br>";
                exit;
            }
            $data = lire_xml($filename);
            print_r($data);

            exit;
        }
    ?>
</body>
</html>