<?php

ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

include ("../utils/connexpdo.inc.php");

function insert_map(string $name, int $nb_nodes, int $nb_arcs) 
{
    echo "insert_map() <br />";
    try {
        $pdo = connexpdo("l3_cc_23_php_map");
        $sql = "
            INSERT INTO MAP(name, num_nodes, num_arcs) 
            VALUES('{$name}',{$nb_nodes},{$nb_arcs})
        ";
        //echo ($pdo->query($sql)) ? "OK" : "booh" ;
        $pdo->query($sql);
        $pdo = null;
    } catch (PDOException $e) {
        displayException($e);
        exit();
    }
}


function insert_nodes(int $map, Array $nodes) 
{
    /*
     * $map   : A map ID
     * $nodes : An array simple containing the node labels.
     * The array index represents the node_num
     */
    echo "insert_nodes() <br />";
    try {
        $pdo = connexpdo("l3_cc_23_php_map");
        $sql = "
            INSERT INTO NODE(map, name, node_num) 
            VALUES(:mapid, :name, :num)
        ";
        $requete = $pdo->prepare($sql);
        $requete->bindParam(':mapid', $map);
        $requete->bindParam(':name', $label);
        $requete->bindParam(':num', $num);

        for($i = 1; $i <= sizeof($nodes); $i++) {
            echo "trying to add {$map}, {$nodes[($i-1)]}, {$i} \t";
            $label = $nodes[($i-1)];
            $num   = $i;
            echo ($requete->execute()) ? "OK<br />" : "booh<br />" ;
        }
        $pdo = null;
    } catch (PDOException $e) {
        displayException($e);
        exit();
    }
}

function insert_arcs(Array $arcs) 
{
    /*
     * $arcs : An associative array containing sub-arrays in the format [head,tail].
     */
    echo "insert_arcs() <br />";
    try {
        $pdo = connexpdo("l3_cc_23_php_map");
        $sql = "
            INSERT INTO ARC(head, tail) 
            VALUES(:head, :tail)
        ";
        $requete = $pdo->prepare($sql);
        $requete->bindParam(':head', $head);
        $requete->bindParam(':tail', $tail);

        foreach($arcs as $a) {
            echo "trying to add arc from {$a['head']} to {$a['tail']} \t";
            $head = $a['head'];
            $tail = $a['tail'];
            echo ($requete->execute()) ? "OK<br />" : "booh<br />" ;
        }
        $pdo = null;
    } catch (PDOException $e) {
        displayException($e);
        exit();
    }
}

// pour tester:
// insert_map("Map2",3,4);
// insert_nodes(2,Array("A","B","C"));
// $arcs = [
//     ['head'=>1, 'tail'=>1],
//     ['head'=>1, 'tail'=>2],
//     ['head'=>2, 'tail'=>3],
//     ['head'=>3, 'tail'=>3]
// ];
// insert_arcs($arcs);

?>