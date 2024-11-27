<?php
    ini_set('display_errors', 1);
    ini_set('display_startup_errors', 1);
    error_reporting(E_ALL);

    include ("../utils/connexpdo.inc.php");

    function select_map(int $map_id) 
    {
        try {
            $pdo = connexpdo("l3_cc_23_php_map");     
            $result = $pdo->query("SELECT * FROM MAP WHERE id={$map_id}");
            $values = $result->fetch(PDO::FETCH_ASSOC);
            //echo "<pre>";print_r($values);echo "</pre>";
            $pdo = null;
            return $values;
        } catch (PDOException $e) {
            displayException($e);
            exit();
        }
    }

    function select_nodes(int $map_id) 
    {
        try {
            $pdo = connexpdo("l3_cc_23_php_map");
            $result = $pdo->query("SELECT * FROM NODE WHERE map={$map_id} ORDER BY node_num ASC");
            $nodes = [];
            while ($row = $result->fetch(PDO::FETCH_ASSOC)) {
                // echo "<pre>";print_r($row);echo "</pre>";
                array_push($nodes, $row['NAME']);
            }
            $pdo = null;
            return $nodes;
        } catch (PDOException $e) {
            displayException($e);
            exit();
        }
    }

    function select_arcs(int $map_id) 
    {
        try {
            $pdo = connexpdo("l3_cc_23_php_map");
            $query = "
                SELECT a.id AS arc, a.head, a.tail, sl.value AS s_label, il.value AS i_label 
                FROM ARC a 
                    LEFT JOIN SYMBOLIC_LABEL sl ON a.id = sl.arc
                    LEFT JOIN INT_LABEL il      ON a.id = il.arc
                WHERE
                    a.head IN (SELECT id FROM NODE where map={$map_id}) OR
                    a.tail IN (SELECT id FROM NODE where map={$map_id})";
            $result = $pdo->query($query);
            $arcs = [];
            while ($row = $result->fetch(PDO::FETCH_ASSOC)) {
                $key = $row['arc'];
                array_splice($row,0,-4);    // remove arc_id
                //echo "<pre>";print_r($row);echo "</pre>";
                $arcs[$key] = $row;
            }
            // echo "<pre>";print_r($arcs);echo "</pre>";
            $pdo = null;
            return $arcs;
        } catch (PDOException $e) {
            displayException($e);
            exit();
        }
    }

    function recuperer_map(int $map_id) 
    {
        $map   = select_map($map_id);
        $nodes = select_nodes($map_id);
        $arcs  = select_arcs($map_id);
        // cast into a Map object or, generate an associative array.
        return Array('map' => $map, 'nodes' => $nodes, 'arcs' => $arcs);
    }

    function afficher_map(Array $map, string $type_label) 
    {
        //echo "<pre>";print_r($map);echo "</pre>";
        echo "{$map['map']['NAME']}<br />"; 
        echo "{$map['map']['NUM_NODES']} nodes<br />"; 
        echo "{$map['map']['NUM_ARCS']} arcs<br />"; 

        // adjacency matrix
        echo "<table>\n";
        echo "<tr>\n<th></th>\n";
        foreach($map['nodes'] as $node => $name) {
            echo "<th>$node ({$name})</th>";
        }
        echo "<tr>\n";
        foreach($map['nodes'] as $row => $rowname) {
            echo "<tr>\n";
            echo "<th>{$row} ({$rowname})</th>";
            foreach($map['nodes'] as $col => $colname) {
                // echo "{$row}, {$col} <br />";
                $arc = array_filter(
                        $map['arcs'], 
                        function($arc) use ($row,$col) { return (($arc['head'] == $row+1) and ($arc['tail'] == $col+1)); }
                    );
                $arc = array_pop($arc);
                $label = ($arc['s_label'] ?? "");
                echo "<td>{$label}</td>";
            }
            echo "<tr>\n";
        }
        echo "</table>\n";
    }
?>

