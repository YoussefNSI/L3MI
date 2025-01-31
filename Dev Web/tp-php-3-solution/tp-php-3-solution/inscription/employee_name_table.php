<?php
$host = $_SERVER['HTTP_HOST'];
$uri = rtrim(dirname($_SERVER['PHP_SELF']), '/\\');

$file = dirname(__FILE__)."/employees.csv";
if (file_exists($file) && $id_file=fopen($file,"r"))
{
    echo "<table border=\"2\"><tbody>";
    echo "<thead><tr><th>NOM</th></tr></thead>";
    while ($tab = fgetcsv($id_file, 200, ";"))
    {
        $id = $tab[0];
        $nom = $tab[1];
        echo "<tr><td><a href=\"http://$host$uri/get_employee.php?id=$id\">$nom</a></td></tr>";
    }
    echo "</tbody></table>";
    fclose($id_file);
}
?>
</body>
</html>
