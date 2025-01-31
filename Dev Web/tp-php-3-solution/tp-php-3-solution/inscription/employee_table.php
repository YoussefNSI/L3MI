<?php
$file = dirname(__FILE__) . "/employees.csv";
$i = 0;
if (file_exists($file) && $id_file = fopen($file, "r")) {
    echo "<table border=\"2\"><tbody>";
    echo "<thead><tr><th>ID</th><th>NOM</th><th>SALAIRE</th><th>AGE</th></tr></thead>";
    while ($tab = fgetcsv($id_file, 200, ";")) {
        $id = $tab[0];
        $nom = $tab[1];
        $salaire = $tab[2];
        $age = $tab[3];
        echo "<tr><td>$id</td><td>$nom</td><td>$salaire</td><td>$age</td></tr>";
        $i++;
    }
    echo "</tbody></table>";
    fclose($id_file);
}
$_SESSION["nb_employees"]=$i;
?>
</body>
</html>
