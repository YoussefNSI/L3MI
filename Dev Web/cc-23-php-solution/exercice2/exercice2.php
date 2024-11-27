<!DOCTYPE html >
<html>
<head>
    <meta charset="UTF-8" />
    <title>Exercice 2</title>
    <link rel="stylesheet" href="../style.css">
</head>

<body>
<?php
    require 'bdd_select.php';
    $mymap = recuperer_map(1);
    afficher_map($mymap,"s");
?>
</body>
</html>

