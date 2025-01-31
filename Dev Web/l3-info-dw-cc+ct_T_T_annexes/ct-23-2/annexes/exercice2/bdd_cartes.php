<?php

// A COMPLETER

function bdd_get_map(int $map_id)
{
}


function bdd_get_arcs(int $map_id)
{

// REQUETE SQL POUR RECUPERER LES ARCS :

//SELECT 
//    a.id AS ARC_NUM, 
//    a.head X_ID, n1.name X_NAME, 
//    a.tail Y_ID, n2.name Y_NAME,
//    sl.value LABEL_STRING, il.value LABEL_INT 
//FROM 
//    ARC a 
//    LEFT JOIN SYMBOLIC_LABEL sl ON a.id = sl.arc
//    LEFT JOIN INT_LABEL il      ON a.id = il.arc
//    INNER JOIN NODE n1          ON a.head = n1.id
//    INNER JOIN NODE n2          ON a.tail = n2.id
//WHERE
//    a.head IN (SELECT id FROM NODE where map=<map_id>) OR
//    a.tail IN (SELECT id FROM NODE where map=<map_id>);
}


function get_arc_label(int $x, int $y, array $arcs)
{
}


function afficher_matrice_adj(int $nb_noeuds, array $arcs)
{
}
