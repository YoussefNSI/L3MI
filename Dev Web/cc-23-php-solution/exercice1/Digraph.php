<?php
declare(strict_types=1);

require_once __DIR__ . "/ADigraph.php";
require_once __DIR__ . "/dfs.php";

/**
 * Digraph est une sous-classe concrète de ADigraph implémentant les digraphes.
 * @see la section DEFINITIONS, NOTATIONS ET TERMINOLOGIE dans le DocBlock de ADigraph.
 * @see la section IMPLEMENTATION dans le DocBlock de ADigraph.
 */
class Digraph extends ADigraph {
    /**
     * Construit le digraphe G à partir de son nombre de noeuds $nrNodes et du tableau $arcs codant son ensemble d'arcs
     * et sa fonction f.
     * 
     * $arcs doit être indexé sur [[E]] et associer chaque clé e dans [[E]] (un identifiant d'arc) à un tableau indexé sur 0..1 
     * qui encode la paire ordonnée de noeuds f(e) :
     * - l'élément d'index 0 encode la queue de e dans la plage [[V]],
     * - l'élément d'index 1 code la tête de e dans la plage [[V]].
     * 
     * $arcs doit être trié lexicographiquement dans l'ordre croissant des identifiants d'arc et des nœuds, 
     * c'est-à-dire,
     * - il est trié dans l'ordre croissant des clés (identifiants d'arcs),
     * - pour deux arcs e_i et e_j tels que f(e_i)=(x_i,y_i), f(e_j)=(x_j,y_j) :
     *      - e_i<e_j implique (x_i<x_j ou (x_i=x_j et y_i<=y_j)).
     * @see la section IMPLEMENTATION dans le DocBlock de l'ADigraph.
     * 
     * @param string $name Le nom de G.
     * @param int $nrNodes Le nombre de noeuds.
     * @param array[] $arcs L'encodage du tableau f.
     * @throws InvalidArgumentException si $nrNodes n'est pas positif.
     * @throws InvalidArgumentException si $arcs n'est pas un encodage valide de f.
     */
    public function __construct(string $name, int $nrNodes, array $arcs)
    {
        $this->name = $name;
        if($nrNodes < 1) {
            throw new InvalidArgumentException("The number of nodes must be positive");
        }
        $this->nrNodes = $nrNodes;
        if(array_keys($arcs) !== range(1, count($arcs))) {
            throw new InvalidArgumentException("The array of arcs must be indexed over range [[E]]");
        }
        foreach($arcs as $arcId => $arc) {
            if(! is_array($arc)) {
                throw new InvalidArgumentException("The encoding of f($arcId) is not an array");
            }            
            if(array_keys($arc) !== [0, 1]) {
                throw new InvalidArgumentException("The encoding of f($arcId) must be an array indexed over range 0..1");
            }
            if(! in_array($arc[0], $this->getNodeIds(), true)) {
                throw new InvalidArgumentException("The tail of f($arcId) is not in range [[V]]");
            }
            if(! in_array($arc[1], $this->getNodeIds(), true)) {
                throw new InvalidArgumentException("The head of f($arcId) is not in range [[V]]");
            }
        }
        $this->arcs = $arcs;
        uasort($arcs, function($a1, $a2) {
                if($a1[0] < $a2[0]) {
                    return -1;
                } elseif($a1[0] == $a2[0]) {
                    return $a1[1] - $a2[1];
                } else {
                    return 1;
                }
        });
        if($arcs != $this->arcs) {
            throw new InvalidArgumentException("The array of arcs must be lexicographically sorted in increasing order of arcs ids and node ids");
        }
    }

    /**
     * Vérifie si le nœud $nodeId appartient à la plage [[V]].
     * 
     * @param int $nodeId Le nœud à vérifier.
     * @return bool true si $nodeId est dans [[V]], false sinon.
     */
    public function checkNodeId(int $nodeId) : bool
    {
        return in_array($nodeId, $this->getNodeIds(), true) ? true : false;
    }

    /**
     * Vérifie si l'identifiant d'arc $arcId appartient à la plage [[E]].
     * 
     * @param int $arcId L'identifiant d'arc à vérifier.
     * @return bool true si $arcId est dans [[E]], false sinon.
     */
    public function checkArcId(int $arcId) : bool
    {
        return array_key_exists($arcId, $this->arcs);
    }

    /**
     * Vérifie si l'arc $arc est bien formé.
     * 
     * $arc est bien formé s'il s'agit d'un tableau d'identifiants de nœuds indexé sur 0..1.
     * @see la section IMPLEMENTATION dans le DocBlock de ADigraph.
     * 
     * @param int[] $arc L'arc à vérifier.
     * @throws InvalidArgumentException Si $arc n'est pas indexé sur 0..1.
     * @throws InvalidArgumentException Si un élément de $arc n'est pas dans [[V]].
     * @return bool true si $arc est bien formé, false sinon.
     */
    public function checkArc(array $arc) : bool
    {
        if(array_keys($arc) !== [0, 1]) {
            //throw new InvalidArgumentException("The array \$arc must be indexed over 0..1.");
            return false;
        }
        if(! is_int($arc[0]) || ! is_int($arc[1])) {
            // throw new InvalidArgumentException("The array \$arc must contain integers.");
            return false;
        }
        if(! $this->checkNodeId($arc[0]) || ! $this->checkNodeId($arc[1])) {
            // throw new InvalidArgumentException("The array \$arc must contain node ids in [[V]]");
            return false;
        }
        return true;
    }

    /**
     * Vérifie si le chemin $path est bien formé et si $simple est vrai, vérifie en plus que le chemin est simple.
     * 
     * $chemin est bien formé si c'est un tableau non vide indexé par des identifiants d'arcs de [[E]], et dont les éléments 
     * encodent des paires ordonnées de noeuds qui forment un chemin valide.
     * @see la section IMPLEMENTATION dans le DocBlock de ADigraph.
     * 
     * @param array[] $path Le chemin à vérifier.
     * @throws InvalidArgumentException Si $path est vide.
     * @param bool $simple Vérifier si le chemin est simple (true) ou non (false).
     * @return bool true si $path est bien formé (et simple si $simple===true), false sinon.
     */
    public function checkDipath(array $path, bool $simple=false) : bool
    {
        if(empty($path)) {
            throw new InvalidArgumentException("A dipath cannot be empty");
        }
        foreach($path as $arcId => $arc) {
            if(! $this->checkArcId($arcId)) {
                return false;
            }
            // if(! is_int($arccId)) {
            //     throw new InvalidArgumentException("Keys of array \$path must be integers");
            //     return false;
            // }
            // if($arcId < 1 || $arcId > $this->getNumberOfNodes()) {
            //     throw new InvalidArgumentException("Arc id must be in [[E]]");
            // }
            if(! $this->checkArc($arc)) {
                return false;
            }
            // if(! is_array($arc)) {
            //     throw new InvalidArgumentException("Values of array \$path must be arrays");
            // }
            // if(array_keys($arc) !== [0, 1] || ! is_int($arc[0]) || ! is_int($arc[1])) {
            //     throw new InvalidArgumentException("Each arc in array \$path must be an array of ints indexed over 0..1");
            // }
            // if(! is_int($arc[0]) || ! is_int($arc[1])) {
            //     throw new InvalidArgumentException("Each arc in array \$path must be an array of ints indexed over 0..1");
            // }
        }
        
        $arcIds = array_keys($path);
        $visitedArcIds= [];
        $visitedNodeIds= [$path[$arcIds[0]][0]];
        for($i=0; $i<count($arcIds)-1; ++$i) {
            if($path[$arcIds[$i]][1] !== $path[$arcIds[$i+1]][0]) {
                return false;
            }
            if(in_array($arcIds[$i], $visitedArcIds)) {
                return false;
            }
            if($simple && in_array($path[$arcIds[$i]][1], $visitedNodeIds)) {
                return false;
            }
            $visitedArcIds[] = $arcIds[$i];
            $visitedNodeIds[] = $path[$arcIds[$i]][1];
        }
        if($simple && $path[$arcIds[0]][0] === $path[$arcIds[count($arcIds)-1]][1]) {
            return false;
        }
        return true;
    }

    /**
     * Retourne le nom de G.
     * 
     * @return string
     */
    public function getName() : string
    {
        return $this->name;
    }

    /**
     * Retourne le nombre de noeuds |V| de G.
     * 
     * @return int
     */
    public function getNumberOfNodes() : int
    {
        return $this->nrNodes;
    }

    /**
     * Retourne le nombre d'arcs |E| de G.
     * 
     * @return int
     */
    public function getNumberOfArcs() : int
    {
        return count($this->arcs);
    }

    /**
     * Retourne le tableau [0 => 1,..|V|-1 => |V|] codant la plage [[V]].
     * @see la section IMPLEMENTATION dans le DocBlock de ADigraph.
     * 
     * @return int[] Tableau indexé d'identifiants de noeuds.
     */
    public function getNodeIds() : array
    {
        return range(1, $this->getNumberOfNodes());
    }

    /**
     * Retourne le tableau [0 => 1,..|E|-1 => |E|] codant la plage [[E]].
     * @see la section IMPLEMENTATION dans le DocBlock de ADigraph.
     * 
     * @return int[] Tableau indexé d'identifiants d'arcs.
     */
    public function getArcIds() : array
    {
        return range(1, $this->getNumberOfArcs());
    }

    /**
     * Retourne l'encodage de f.
     * @see la section IMPLEMENTATION dans le DocBlock de l'ADigraph.
     * 
     * @return array[] tableau indexé de tableaux encodant f.
     */
    public function getArcs() : array
    {
        return $this->arcs;
    }

    /**
     * Retourne f($arcId) où $arcId est l'identifiant d'un arc.
     * @see la section IMPLEMENTATION dans le DocBlock de ADigraph.
     * 
     * @param int $arcId L'identifiant de l'arc.
     * @throws InvalidArgumentException si $arcId n'est pas dans [[E]].
     * @return int[] Tableau d'entiers indexé sur 0..1 encodant f($arcId).
     */
    public function getArc(int $arcId) : array
    {
        if(! $this->checkArcId($arcId)) {
            throw new InvalidArgumentException("The arc id must be in range 1.." . $this->getNumberOfArcs());
        }
        return $this->arcs[$arcId];
    }

    /**
     * Retourne la restriction de f aux arcs entrants du noeud $nodeId.
     * L'extraction préserve les index des arcs dans f et leur ordre.
     * @see la section IMPLEMENTATION dans le DocBlock de l'ADigraph.
     * 
     * @param int $nodeId L'identifiant du noeud.
     * @throws InvalidArgumentException si $nodeId n'est pas dans [[V]].
     * @return array[] Tableau indexé de tableaux codant les arcs entrants du noeud.
     */
    public function getIncomingArcs(int $nodeId) : array
    {
        if(! $this->checkNodeId($nodeId)) {
            throw new InvalidArgumentException("The node id must be in range 1.." . $this->getNumberOfNodes());
        }
        $arcs = array_filter($this->arcs, function($arc) use($nodeId) {
            return $arc[1] == $nodeId;
        });
        
        return $arcs;
    }

    /**
     * Retourne la restriction de f aux arcs sortants du noeud $nodeId.
     * L'extraction préserve les index des arcs dans f et leur ordre.
     * @see la section IMPLEMENTATION dans le DocBlock de l'ADigraph.
     * 
     * @param int $nodeId L'identifiant du noeud.
     * @throws InvalidArgumentException si $nodeId n'est pas dans [[V]].
     * @return array[] Tableau indexé de tableaux codant les arcs sortants du noeud.
     */
    public function getOutgoingArcs(int $nodeId) : array
    {
        if(! $this->checkNodeId($nodeId)) {
            throw new InvalidArgumentException("The node id must be in range 1.." . $this->getNumberOfNodes());
        }
        $arcs = array_filter($this->arcs, function($arc) use($nodeId) {
            return $arc[0] == $nodeId;
        });
        
        return $arcs;
    }

    /**
     * Retourne le tableau des prédécesseurs directs du nœud $nodeId triés par ordre croissant.
     * 
     * @param int $nodeId L'identifiant du nœud.
     * @throws InvalidArgumentException si $nodeId n'est pas dans [[V]].
     * @return int[] Tableau indexé de nœuds triés par ordre croissant.
     */
    public function getDirectPredecessors(int $nodeId) : array
    {
        try {
            $arcs = $this->getIncomingArcs($nodeId);
            $predecessors = array_map(fn($arc) => $arc[0], $arcs);
            sort($predecessors);
            return $predecessors;
        } catch(InvalidArgumentException $e) {
            throw $e;
        }
    }

    /**
     * Retourne le tableau des successeurs directs du nœud $nodeId triés par ordre croissant.
     * 
     * @param int $nodeId L'identifiant du nœud.
     * @throws InvalidArgumentException si $nodeId n'est pas dans [[V]].
     * @return int[] Tableau indexé de nœuds triés par ordre croissant.
     */
    public function getDirectSuccessors(int $nodeId) : array
    {
        try {
            $arcs = $this->getOutgoingArcs($nodeId);        
            $successors = array_map(fn($arc) => $arc[1], $arcs);
            sort($successors);
            return $successors;
        } catch(InvalidArgumentException $e) {
            throw $e;
        }
    }

    /**
     * Retourne le tableau des nœuds atteignables à partir du nœud $nodeId, triés par ordre croissant.
     * 
     * @param int $nodeId L'identifiant du nœud.
     * @throws InvalidArgumentException si $nodeId n'est pas dans [[V]].
     * @return int[] Tableau indexé de nœuds triés par ordre croissant.
     */
    public function getReachableNodes(int $nodeId) : array
    {
        if(! $this->checkNodeId($nodeId)) {
            throw new InvalidArgumentException("The node id must be in range 1.." . $this->getNumberOfNodes());
        }
        $reachables = [];
        foreach($this->getNodeIds() as $destination) {
            $paths = $this->getDipaths($nodeId, $destination, false);
            if($paths !== []) {
                $reachables[] = $destination; 
            }
        }
        return $reachables;
    }

    /**
     * Retourne le tableau des nœuds sources de G triés par ordre croissant.
     * 
     * @return int[] Tableau indexé de nœuds triés par ordre croissant.
     */
    public function getSourceNodes() : array
    {
        $sources = array_filter($this->getNodeIds(), function($node) {
            return $this->getDirectPredecessors($node) === [];
        });
        
        return $sources;
    }

    /**
     * Retourne le tableau des nœuds sources dans G, triés par ordre croissant.
     * 
     * @return int[] Tableau indexé de nœuds triés par ordre croissant.
     */
    public function getSinkNodes() : array
    {
        $sinks = array_filter($this->getNodeIds(), function($node) {
            return $this->getDirectSuccessors($node) === [];
        });
        
        return $sinks;
    }

    /**
     * Retourne le tableau des chemins allant du noeud origine $origin au noeud destination $destination
     * si $simple est faux, sinon retourne le tableau des chemins simples allant de $origin à $destination.
     * 
     * Le tableau est indexé numériquement de 0 au nombre de chemins (ou chemins simples).
     * Il est vide si aucun chemin n'existe (ou aucun chemin simple n'existe) entre $origin et $destination.
     * Voir la section IMPLEMENTATION dans le DocBlock d'ADigraph.
     * 
     * @param int $origin L'identifiant du noeud d'origine.
     * @param int $destination L'identifiant du nœud de destination.
     * @param bool $simple Indique s'il faut extraire uniquement les chemins simples (true) ou tous les chemins (false).
     * @throws InvalidArgumentException si l'identifiant de l'origine ou de la destination n'est pas dans [[V]].
     * @throws InvalidArgumentException si $simple est vrai et que les nœuds d'origine et de destination sont les mêmes.
     * @return array[] Tableau indexé de tableaux encodant les chemins (ou chemins simples).
     */
    public function getDipaths(int $origin, int $destination, bool $simple=false) : array
    {
        if(! $this->checkNodeId($origin)) {
            throw new InvalidArgumentException("The origin node must be in range 1.." . $this->getNumberOfNodes());
        }
        if(! $this->checkNodeId($destination)) {
            throw new InvalidArgumentException("The destination node must be in range 1.." . $this->getNumberOfNodes());
        }
        if($simple && $origin === $destination) {
            throw new InvalidArgumentException("Simple paths can never go from a node to itself");
        }
        $path = [];
        $paths = [];
        DFS($this, $origin, $destination, $path, $paths, $simple);
        return $paths;
    }

    /**
     * Retourne le tableau de tous les chemins du graphe si $simple est faux,
     * sinon renvoie le tableau de tous les chemins simples du graphe.
     * @voir la section IMPLEMENTATION dans le DocBlock de ADigraph.
     * 
     * @param bool $simple Indique s'il faut extraire tous les chemins simples (true) ou tous les chemins (false).
     * @return array[] Tableau indexé de tableaux codant tous les chemins dans G (ou tous les chemins simples dans G).
     */
    public function getAllPaths(bool $simple=false) : array
    {
        $allPaths = [];
        foreach($this->getNodeIds() as $origin) {
            foreach($this->getNodeIds() as $destination) {
                if(! $simple || $origin !== $destination) {
                    $paths = $this->getDipaths($origin, $destination, $simple);
                    $allPaths = array_merge($allPaths, $paths);
                    }
            }
        }
        return $allPaths;
    }

    /**
     * Retourne le degré sortant du graphe.
     * 
     * @return int Le degré de sortie maximal des nœuds.
     */
    public function getMaxOutDegree() : int
    {
        return max(array_map(
                    fn($node) => count($this->getDirectSuccessors($node)),
                    $this->getNodeIds())
                );
    }

    /**
     * Retourne vrai si G est un digraphe simple.
     * 
     * @return bool
     */
    public function isSimpleDigraph() : bool
    {
        foreach($this->getArcs() as $arc) {
            if($arc[0] === $arc[1]) {
                return false;
            }
        }
        foreach($this->getArcs() as $arcId1 => $arc1) {
            foreach($this->getArcs() as $arcId2 => $arc2) {
                if($arcId1 < $arcId2) {
                    if($arc1[0] === $arc2[1] && $arc1[1] === $arc2[1]) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * Retourne vrai si G est cyclique.
     * 
     * @return bool Si G est cyclique (true) ou non (false).
     */
    public function isCyclic() : bool
    {
        foreach($this->getNodeIds() as $i) {
            foreach($this->getReachableNodes($i) as $j) {
                if($i<$j) {
                    if(in_array($i, $this->getReachableNodes($j), true)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
     * Retourne vrai si G est fortement connecté.
     * 
     * @return bool Si G est fortement connecté (true) ou non (false).
     */
    public function isStronglyConnected() : bool {
        foreach($this->getNodeIds() as $i) {
            if($this->getReachableNodes($i) != $this->getNodeIds()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Construit et retourne un objet implémentant ADigraph à partir du fichier JSON $jsonFile.
     * 
     * @return ADigraph
     */
    static public function fromJson(string $jsonFile) : ADigraph
    {
        $jsonMap = json_decode(file_get_contents($jsonFile), true);
        $name = $jsonMap["name"];
        $nrNodes = $jsonMap["digraph"]["nrNodes"];
        $arcs = $jsonMap["digraph"]["arcs"];
        $arcs = array_combine(range(1, count($arcs)), $arcs);
        return new Digraph($name, $nrNodes, $arcs);
    }

    /**
     * Encode et retourne G au format JSON.
     * 
     * @return string
     */
    public function toJson() : string
    {
        return
            json_encode([
                    "nrNodes" => $this->getNumberOfNodes(),
                    "nrArcs" => $this->getNumberOfArcs(),
                    "arcs" => $this->arcs
                ],
                JSON_FORCE_OBJECT
            );
    }

    /**
     * Encode et retourne les indicateurs de base de G sous forme de chaîne de caractères et si $showName est vrai, encode également le nom du graphe.
     * 
     * @param bool $showName Encode le nom du graphe (true) ou non (false)
     * @return string
     */
    public function coreToString(bool $showName=false) : string {
        $sources = $this->getSourceNodes();
        $sinks = $this->getSinkNodes();
        return 
            ($showName ? "\nDIGRAPH $this->name" : "") .
            "\n- #nodes: " . $this->getNumberOfNodes() .
            "\n- #arcs: " . $this->getNumberOfArcs() .
            "\n- simple? " . ($this->isSimpleDigraph() ? "true" : "false") .
            "\n- #sources: " . implode(" ,", array_map(fn($i) => "$i", $sources)) .
            "\n- #sinks: " . implode(" ,", array_map(fn($i) => "$i", $sinks)) .
            "\n- #max out-degree: " . $this->getMaxOutDegree();
    }

    /**
     * Encode et retourne le tableau indexé d'arcs $arcs sous forme de chaîne de caractères.
     * 
     * @param array[] $arcs Tableau indexé de tableaux.
     * @return string
     */
    public function arcsToString(array $arcs) : string
    {
        $str 
            = implode(
                ",",
                array_map(
                    function($arcId, $arc) {
                        return "$arcId => (" . $arc[0] . "," . $arc[1] . ")";
                    },
                    array_keys($arcs),
                    array_values($arcs)),
                );
        return $str;
    }

    /**
     * Encode et retourne le tableau de chemins $paths sous forme de chaîne de caractères.
     * 
     * @param array[] $paths Tableau indexé de tableaux de tableaux.
     * @return string
     */
    public function pathsToString(array $paths) : string
    {
        $str 
            = implode(
                "\n",
                array_map(
                    function($path) {
                        $origin = reset($path)[0];
                        $destination = end($path)[1];
                        reset($path);
                        return
                            $origin . "->" . $destination . ": " . $this->arcsToString($path);
                    },
                    $paths)
                )
                . "\n";
        return $str;
    }

    /**
     * Encode et retourne les propriétés clés de G sous forme de chaîne de caractères.
     * Si $showName est vrai, le nom du graphe est également encodé.
     * Si $showProperties est vrai, les propriétés de G sont également encodées.
     * Si $showDipaths est vrai, la liste des chemins est également encodée.
     * Si $showSimpleDipaths est vrai, la liste des chemins simples est également encodée.
     * 
     * @param bool $showName Encoder ou non (true) le nom du graphe (false)
     * @param bool $showProperties Encoder ou non (true) les propriétés du graphe (false)
     * @param bool $showDipaths Encode la liste des chemins (true) ou non (false)
     * @param bool $showSimpleDipaths Indique s'il faut encoder la liste des chemins simples (true) ou non (false)
     * @return string
     */
    public function propertiesToString(bool $showName=false, bool $showProperties=false, bool $showDipaths=false, bool $showSimpleDipaths=false) : string {
        $allPaths = $this->getAllPaths(false);
        $allSimplePaths = $this->getAllPaths(true);
        return 
            ($showName ? "\nDIGRAPH $this->name" : "") .
            ($showProperties ?
                ("- cyclic? " . ($this->isCyclic() ? "true" : "false") .
                "\n- strongly-connected? " . ($this->isStronglyConnected() ? "true" : "false") .
                "\n- #paths: " . count($allPaths) .
                "\n- #simple-paths: " . count($allSimplePaths))
                : "") .
            ($showDipaths ? "\nDIPATHS:\n" . $this->pathsToString($allPaths) : "") .
            ($showSimpleDipaths ? "\nSIMPLE DIPATHS:\n" . $this->pathsToString($allSimplePaths) : "") .
            "\n";
    }

    /**
     * Encode et retourne les éléments clés et les propriétés de G sous forme de chaîne de caractères.
     * 
     * @return string
     */
    public function __toString() : string {
        return 
            $this->coreToString(true) .
            $this->propertiesToString(false, false, false, false) .
            "\n";
    }
}
?>