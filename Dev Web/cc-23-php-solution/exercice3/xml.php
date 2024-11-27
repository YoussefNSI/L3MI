<?php
    ini_set('display_errors', 1);
    ini_set('display_startup_errors', 1);
    error_reporting(E_ALL);

    function lire_xml(string $filename) 
    {
        /**
         * lire_xml()
         * Lecture d'une carte cognitive en format XML.
         * 
         * $filename : Nom du fichier à lire
         * 
         * Returns:
         * Un tableau associatif avec les données récupèrés.
         */

        $data = [];
        $data['name'] = $filename;

        $xml = simplexml_load_file($filename);
        // echo "<pre>";
        // print_r($xml);
        // echo "</pre><br /><br />";

        $nb_nodes = $nb_arcs = 0;
        $data['nb_nodes'] = $nb_nodes = (int) $xml->graph->size['nodes'];
        $data['nb_arcs']  = $nb_arcs  = (int) $xml->graph->size['arcs'];
        // echo "{$nb_nodes} nodes, {$nb_arcs} arcs.<br /><br />";
        
        // arcs -------------------------------------
        $arcs = [];
        $arc_id = $arc_head = $arc_tail = 0;
        foreach($xml->graph->arcs->arc as $arc) {
            $arc_id   = (int) $arc['id'];
            $arc_head = (int) $arc['head'];
            $arc_tail = (int) $arc['tail'];
            // echo "{$arc_id} : {$arc_head} - {$arc_tail} <br />"; 
            $arcs[$arc_id] = ['head'=>$arc_head, 'tail'=>$arc_tail];
        }
        $data['arcs'] = $arcs;

        // node labels ------------------------------
        $node_labels = [];
        foreach($xml->labeling->nodes->node as $node) {
            array_push($node_labels, (string)$node);
        }
        // print_r($node_labels);
        // echo "<br /><br />";
        $data['nodes'] = $node_labels;

        // arc labels -------------------------------
        $arc_labels_type = $xml->labeling->influence_domain['type'];

        $arc_labels = [];
        if ($arc_labels_type == "integer") {
            foreach($xml->labeling->labels->int_label as $label) {
                array_push($arc_labels, (int)$label['value']);
            }
        } elseif ($arc_labels_type == "symbolic") {
            foreach($xml->labeling->labels->symbolic_label as $label) {
                array_push($arc_labels, (string)$label['value']);
            }
        }
        // print_r($arc_labels); echo "<br /><br />";
        $data['arc_labels'] = $arc_labels;

        return $data;
    }
?>