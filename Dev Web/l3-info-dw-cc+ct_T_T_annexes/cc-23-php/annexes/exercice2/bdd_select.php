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
?>

