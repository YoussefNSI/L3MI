<?php
require_once __DIR__ . "/Digraph.php";

$jsonFiles[] = __DIR__ . "/data/map_3_3_acyclic.json";
$jsonFiles[] = __DIR__ . "/data/map_3_3_cyclic.json";
$jsonFiles[] = __DIR__ . "/data/map_8_8_driving.json";

foreach($jsonFiles as $jsonFile) {
    $digraph = Digraph::fromJson($jsonFile);
    echo $digraph;
    echo $digraph->propertiesToString(false, true, true, true);
}

/* RESULTAT ATTENDU EN CONSOLE

[exercice1]$ php test.php
DIGRAPH map_3_3_acyclic
- #nodes: 3
- #arcs: 3
- simple? true
- #sources: 1
- #sinks: 3
- #max out-degree: 2

- cyclic? false
- strongly-connected? false
- #paths: 4
- #simple-paths: 4
DIPATHS:
1->2: 1 => (1,2)
1->3: 1 => (1,2),3 => (2,3)
1->3: 2 => (1,3)
2->3: 3 => (2,3)

SIMPLE DIPATHS:
1->2: 1 => (1,2)
1->3: 1 => (1,2),3 => (2,3)
1->3: 2 => (1,3)
2->3: 3 => (2,3)


DIGRAPH map_3_3_cyclic
- #nodes: 3
- #arcs: 3
- simple? true
- #sources: 
- #sinks: 
- #max out-degree: 1

- cyclic? true
- strongly-connected? true
- #paths: 9
- #simple-paths: 6
DIPATHS:
1->1: 1 => (1,2),2 => (2,3),3 => (3,1)
1->2: 1 => (1,2)
1->3: 1 => (1,2),2 => (2,3)
2->1: 2 => (2,3),3 => (3,1)
2->2: 2 => (2,3),3 => (3,1),1 => (1,2)
2->3: 2 => (2,3)
3->1: 3 => (3,1)
3->2: 3 => (3,1),1 => (1,2)
3->3: 3 => (3,1),1 => (1,2),2 => (2,3)

SIMPLE DIPATHS:
1->2: 1 => (1,2)
1->3: 1 => (1,2),2 => (2,3)
2->1: 2 => (2,3),3 => (3,1)
2->3: 2 => (2,3)
3->1: 3 => (3,1)
3->2: 3 => (3,1),1 => (1,2)


DIGRAPH map_8_8_driving
- #nodes: 8
- #arcs: 8
- simple? true
- #sources: 1 ,2 ,3 ,4
- #sinks: 8
- #max out-degree: 2

- cyclic? false
- strongly-connected? false
- #paths: 13
- #simple-paths: 13
DIPATHS:
1->5: 1 => (1,5)
1->8: 1 => (1,5),6 => (5,8)
2->5: 2 => (2,5)
2->6: 3 => (2,6)
2->8: 2 => (2,5),6 => (5,8)
2->8: 3 => (2,6),7 => (6,8)
3->7: 4 => (3,7)
3->8: 4 => (3,7),8 => (7,8)
4->7: 5 => (4,7)
4->8: 5 => (4,7),8 => (7,8)
5->8: 6 => (5,8)
6->8: 7 => (6,8)
7->8: 8 => (7,8)

SIMPLE DIPATHS:
1->5: 1 => (1,5)
1->8: 1 => (1,5),6 => (5,8)
2->5: 2 => (2,5)
2->6: 3 => (2,6)
2->8: 2 => (2,5),6 => (5,8)
2->8: 3 => (2,6),7 => (6,8)
3->7: 4 => (3,7)
3->8: 4 => (3,7),8 => (7,8)
4->7: 5 => (4,7)
4->8: 5 => (4,7),8 => (7,8)
5->8: 6 => (5,8)
6->8: 7 => (6,8)
7->8: 8 => (7,8)

[exercice1]$
/**/
?>