<?php
    /**
     * module 'bdd_select'
     * 
     * function select_map(...)
     * function select_nodes(...) 
     * function select_arcs(...) 
     * 
     * Pour récupèrer les arcs dans une carte, en utilisant 
     * seulement son identifiant (<map_id>), utilisez la query suivante:
     * 
        SELECT a.id AS arc, a.head, a.tail, sl.value AS s_label, il.value AS i_label 
        FROM ARC a 
            LEFT JOIN SYMBOLIC_LABEL sl ON a.id = sl.arc
            LEFT JOIN INT_LABEL il      ON a.id = il.arc
        WHERE
            a.head IN (SELECT id FROM NODE where map=<map_id>) OR
            a.tail IN (SELECT id FROM NODE where map=<map_id>)";
     */

    include ("../utils/connexpdo.inc.php");

    function select_map($map_id){
        try {
            $objdb = connexpdo("l3_cc_23_php_map");
            $str = "SELECT * FROM map WHERE id = $map_id";
            $result = $objdb->query($str);
            $values = $result->fetch(PDO::FETCH_ASSOC);
            $objdb = null;
            return $values;
        } catch (PDOException $e) {
            echo "erreur";
            displayException($e);
        }
    }

    function select_nodes($map_id){
        try {
            $objdb = connexpdo("l3_cc_23_php_map");
            $str = "SELECT * FROM node WHERE map = $map_id ORDER BY node_num ASC";
            $result = $objdb->query($str);
            $nodes = [];
            while($values = $result->fetch(PDO::FETCH_ASSOC)){
                array_push($nodes, $values['NAME']);
            }
            $objdb = null;
            return $values;
        } catch (PDOException $e) {
            echo "erreur";
            displayException($e);
        }
    }

    function select_arcs($map_id){
        try {
            $objdb = connexpdo("l3_cc_23_php_map");
            $str = "SELECT a.id AS arc, a.head, a.tail, sl.value AS s_label, il.value AS i_label 
                    FROM ARC a 
                        LEFT JOIN SYMBOLIC_LABEL sl ON a.id = sl.arc
                        LEFT JOIN INT_LABEL il      ON a.id = il.arc
                    WHERE
                        a.head IN (SELECT id FROM NODE where map=$map_id) OR
                        a.tail IN (SELECT id FROM NODE where map=$map_id)";
            $result = $objdb->query($str);
            while($values = $result->fetch(PDO::FETCH_ASSOC)){
                var_dump($values);
                $key = $values['arc'];
                array_splice($values, 0, -4);
                $arcs[$key] = $values;
            }
            $objdb = null;
            return $values;
        } catch (PDOException $e) {
            echo "erreur";
            displayException($e);
        }
    }
?>

