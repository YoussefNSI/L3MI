<?php
//print_r($_POST);

$facettes = ["athlète", "sport", "période", "médailles"];
$solution = [
    "alain" => [
        $facettes[1] => "course",
        $facettes[2] => "2004",
        $facettes[3] => "or"
    ],
    "pierre" => [
        $facettes[1] => "natation",
        $facettes[2] => "1988",
        $facettes[3] => "argent"
    ],    
    "maurice" => [
        $facettes[1] => "aviron",
        $facettes[2] => "2016",
        $facettes[3] => "bronze"
    ]
];
$reponse = [];
$resolu = 1;

forEach($solution as $k1 => $tab) {
    forEach($tab as $k2 => $v2) {
        $key = $k1 . "_" . $k2;
        if (array_key_exists($key, $_POST)) {
            $reponse[$key] = (int) ($_POST[$key] === $v2);
            $resolu *= $reponse[$key];
        } else {
            echo "Clé du menu $key non communiquée !";
            exit();
        }
    }
}

echo json_encode(["résolution" => $resolu, "réponse" => $reponse]);
?>
