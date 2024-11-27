<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="../style.css">
    <title>2023 CC-PHP Exercice 3</title>
    <!--style type="text/css">
        * { font-family: 'Roboto', sans-serif; text-align: center;}
        .error { color: red }
        select, input { padding: 3px; }
    </style -->
</head>
<body>
    <form enctype="multipart/form-data" method="post" action="<?= $_SERVER['PHP_SELF'];?>" >
        <label for="fichier">XML à importer:
            <input type="file" name="fichier" id="fichier" accept=".xml">
        </label>
        <input type="submit" name="submit" value="Ajouter à la base de données">
    </form>

    <?php
        // A COMPLETER
        include ('xml.php');
        include ('bdd_insert.php');

        if (isset($_FILES["fichier"])) 
        {
            $filename = "fichier.xml";
            $uploadfile = __DIR__ . "/$filename";
            if (move_uploaded_file($_FILES['fichier']['tmp_name'], $uploadfile)) {
                echo "File is valid, and was successfully uploaded.\n";
            } else {
                exit;
            }
            $data = lire_xml($filename);
            print_r($data);
            
            exit;

            // version 'tableau associatif'
            insert_map("Map2",3,4);
            insert_nodes(2,Array("A","B","C"));
            $arcs = [
                ['head'=>1, 'tail'=>1],
                ['head'=>1, 'tail'=>2],
                ['head'=>2, 'tail'=>3],
                ['head'=>3, 'tail'=>3]
            ];
            insert_arcs($arcs);

        }
    ?>
</body>
</html>