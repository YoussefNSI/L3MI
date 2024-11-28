<?php
    /**
     * module 'xml.php'
     * 
     * function lire_xml(...) 
     */

     function lire_xml($filename) {
        $data = [];
        $data['name'] = $filename;

        $xml = simplexml_load_file($filename);

        $nb_nodes = 0;
        $nb_arcs = 0;

        $data['nb_nodes'] = $nb_nodes = (int) $xml->graph->size['nodes'];
        $data['nb_arcs'] = $nb_arcs = (int) $xml->graph->size['arcs'];

        $arcs = [];
        $arc_id = $arc_tail = $arc_head = 0;
        foreach($xml->graph->arcs as $arc){
            $arc_id = (int) $arc['id'];
            $arc_tail = (int) $arc['tail'];
            $arc_head = (int) $arc['head'];
            $arcs[$arc_id] = ['head' => $arc_head, 'tail' => $arc_tail];
        }
        $data['arcs'] = $arcs;

        $nodes = [];
        $node_num = 0;
        $node_name = "";
        foreach($xml->graph->nodes as $node){
            $node_num = (int) $node['id'];
            $node_name = (string) $node['name'];
            $nodes[$node_num] = ['name' => $node_name];
        }
        $data['nodes'] = $nodes;

        $arc_labels_type = $xml->labeling->influence_domain['type'];
        $arc_labels = [];

        if($arc_labels_type == "integer"){
            foreach($xml->labeling->labels as $label){
                array_push($arc_labels, (int) $label['value']);
            }
        }
        elseif($arc_labels_type == "symbolic"){
            foreach($xml->labeling->labels as $label){
                array_push($arc_labels, (string) $label['value']);
            }
        }

        $data['arc_labels'] = $arc_labels;

        return $data;
     }

?>