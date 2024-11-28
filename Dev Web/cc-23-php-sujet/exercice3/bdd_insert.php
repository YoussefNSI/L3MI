<?php
    /**
     * module 'bdd_insert'
     * 
     * function insert_map(...)
     * function insert_nodes(...) 
     * function insert_arcs(...) 
     */

    require_once '../utils/connexpdo.inc.php';

    function insert_map(string $mapname, int $nb_nodes, int $nb_arcs){
        echo "insert_map() <br />";
        try{
            $pdo = connexpdo("l3_cc_23_php_map");
            $str = "INSERT INTO MAP(name, num_nodes, num_arcs) VALUES($mapname, $nb_nodes, $nb_arcs)";
            $pdo->query($str);
            $pdo = null;
        }
        catch(PDOException $e){
            echo $e->getMessage();
            exit();
        }
    }

    function insert_nodes(int $map, Array $nodes){
        try{
            $pdo = connexpdo("l3_cc_23_php_map");
            $str = "
                INSERT INTO NODE(map, name, node_num)
                VALUES(:mapid, :name, :num)";
            $stmt = $pdo->prepare($str);
            $stmt->bindParam(':mapid', $map);
            $stmt->bindParam(':name', $label);
            $stmt->bindParam(':num', $num);

            for($i = 1; $i <= sizeof($nodes); $i++){
                echo "trying to add {$map}, {$nodes[($i-1)]}, {$i} \t";
                $label = $nodes[($i-1)];
                $num = $i;
                echo ($stmt->execute()) ? "OK<br />" : "booh<br />" ;
            }
            $pdo = null;
        }
        catch(PDOException $e){
            echo $e->getMessage();
            exit();
        }
    }

    function insert_arcs(int $map, Array $arcs){
        try{
            $pdo = connexpdo("l3_cc_23_php_map");
            $str = "
                INSERT INTO ARC(head, tail)
                VALUES(:head, :tail)";
            $stmt = $pdo->prepare($str);
            $stmt->bindParam(':head', $head);
            $stmt->bindParam(':tail', $tail);

            foreach($arcs as $arc){
                echo "trying to add {$arc[0]}, {$arc[1]} \t";
                $head = $arc['head'];
                $tail = $arc['tail'];
                echo ($stmt->execute()) ? "OK<br />" : "booh<br />" ;
            }
            $pdo = null;
        }
        catch(PDOException $e){
            echo $e->getMessage();
            exit();
        }
    }
?>