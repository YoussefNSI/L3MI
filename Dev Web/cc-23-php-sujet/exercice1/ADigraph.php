<?php
declare(strict_types=1);

/**
 * ADigraph est la classe abstraite implémentant un digraphe.
 * 
 * I. DÉFINITIONS, NOTATIONS ET TERMINOLOGIE.
 * 
 * Un *digraphe* G est un triplet (V,E,f) tel que :
 * - V est un ensemble de *nœuds*,
 * - E est un ensemble d'*arcs*,
 * - f : E -> VxV est une fonction qui fait correspondre chaque arc à une paire ordonnée de nœuds.
 * 
 * Soit e un arc de G tel que f(e)=(x,y) :
 * - x est appelé la *queue* de e.
 * - y est appelé la *tête* de e.
 * - e est appelé *arc sortant* de x.
 * - e est appelé *arc entrant* de y.
 * 
 * Un digraphe est dit *simple* s'il n'a ni boucles, ni arc dupliqué :
 * - [pas de boucles] pour e dans E, v dans V, f(e)!=(v,v).
 * - [pas d'arcs dupliqués] pour e_i, e_j dans E, e_i!=e_j implique f(e_i)!=f(e_j).
 * 
 * Un *chemin* est une séquence finie (e_1,...,e_m) de m arcs de G (m>=0) telle que :
 * - [jointure de noeuds] pour i=1...m-1, la tête de e_i est la queue de e_{i+1},
 * - [arcs différents] pour 1<=i<j<=m, e_i!=e_j.
 * 
 * La *séquence de noeuds* d'un chemin p=(e_1,...,e_m) est la séquence de noeuds (v_1,...,v_{m+1}) telle que f(e_i)=(v_i,v_{i+1}) pour i=1...m :
 * - p est appelé chemin de v_1 à v_{m+1}
 * - v_1 est appelé *origine* de p
 * - v_{m+1} est appelé *destination* de p.
 * 
 * Un chemin de séquence de noeuds (v_1,...,v_{n}) est dit *chemin simple* si
 * - [nœuds distincts] pour 1<=i<j<=n, v_i!=v_j.
 * 
 * Soient x, y deux noeuds de G (potentiellement identiques, x=y) :
 * - x est un *prédécesseur direct* de y s'il existe e dans E tel que f(e)=(x,y).
 * - y est un *successeur direct* de x s'il existe e dans E tel que f(e)=(x,y).
 * - y est *atteignable* à partir de x s'il existe un chemin de x à y.
 * 
 * Un digraphe est dit *cyclique* s'il existe des noeuds x et y tels que :
 * - x est atteignable à partir de y.
 * - y est atteignable à partir de x.
 * 
 * Un digraphe est dit *fortement connecté* si chaque noeud est atteignable à partir de n'importe quel autre y compris lui-même.
 * 
 * Soit x un noeud de G :
 * - x est dit *source* de G s'il n'a pas d'arcs entrants.
 * - x est dit *puits* de G s'il n'a pas d'arcs sortants.
 * - Le *degré entrant* de x est le nombre de ses arcs entrants.
 * - Le *degré sortant* de x est le nombre de ses arcs sortants.
 * - Le *degré entrant* de G est le degré entrant maximum de ses noeuds.
 * - Le *degré sortant* de G est le degré sortant maximum de ses noeuds.
 * 
 * 
 * II. IMPLEMENTATION
 * 
 * Une *plage* est un ensemble d'entiers consécutifs.
 * - Pour tous les entiers m,n tels que m<n, la plage {i | m<=i<=n} est notée m..n.
 * 
 * Soit X un ensemble fini d'entiers.
 * - |X| désigne la cardinalité de X.
 * - [[X]] désigne la plage 1..|X|.
 * 
 * Les nœuds sont codés (identifiés) par des entiers dans la plage [[V]], c'est-à-dire que V est codé par le tableau [1,2, ...|V|].
 * 
 * Les arcs sont codés (identifiés) par des entiers dans la plage [[E]], c'est-à-dire que E est codé par le tableau [1,2, ...|E|].
 * 
 * Chaque paire ordonnée de nœuds f(e) associée par f à un arc e est codée par un tableau de nœuds indexé sur 0..1 :
 * - 0 est l'index du noeud de queue
 * - 1 est l'index du nœud de tête.
 *  
 * f est encodée par un tableau indexé sur [[E]] qui 
 * - associe chaque clé e dans E (un identifiant d'arc) à un tableau codant la paire ordonnée de nœuds f(e) (voir ci-dessus),
 * - est trié lexicographiquement dans l'ordre croissant des identifiants d'arc et des nœuds :
 *      - il est trié dans l'ordre croissant des clés (identifiants d'arcs),
 *      - pour tous les e_i, e_j dans [[E]] tels que f(e_i)=(x_i,y_i) et f(e_j)=(x_j,y_j) : e_i<e_j implique (x_i<x_j ou (x_i=x_j et y_i<=y_j)).
 * 
 * Chaque chemin (simple ou non) est codé par un tableau associatif tel que :
 * - la n-ième clé est l'identifiant du n-ième arc du chemin
 * - la valeur associée à la clé e (l'identifiant d'un arc) est l'encodage de f(e).
 */

abstract class ADigraph {
    /**
     * Le nom de G.
     * 
     * @var string
     */
    protected $name ;

    /**
     * Le nombre de nœuds |V| de G.
     * 
     * Les nœuds sont codés sur la plage [[V]].
     *  
     * @var int
     */
    protected int $nrNodes ;

    /**
     * La fonction d'encodage du tableau f de G.
     * @see la section IMPLEMENTATION dans le DocBlock de ADigraph.
     * 
     * @var array[]
     */
    protected $arcs ;

    /**
     * Vérifie si le nœud $nodeId appartient à la plage [[V]].
     * 
     * @param int $nodeId Le nœud à vérifier.
     * @return bool true si $nodeId est dans [[V]], false sinon.
     */
    abstract public function checkNodeId(int $nodeId) : bool ;

    /**
     * Vérifie si l'identifiant d'arc $arcId appartient à la plage [[E]].
     * 
     * @param int $arcId L'identifiant d'arc à vérifier.
     * @return bool true si $arcId est dans [[E]], false sinon.
     */
    abstract public function checkArcId(int $arcId) : bool ;

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
    abstract public function checkArc(array $arc) : bool ;

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
    abstract public function checkDipath(array $path, bool $simple=false) : bool ;

    /**
     * Retourne le nom de G.
     * 
     * @return string
     */
    abstract public function getName() : string ;

    /**
     * Retourne le nombre de noeuds |V| de G.
     * 
     * @return int
     */
    abstract public function getNumberOfNodes() : int ;

    /**
     * Retourne le nombre d'arcs |E| de G.
     * 
     * @return int
     */
    abstract public function getNumberOfArcs() : int ;

    /**
     * Retourne le tableau [0 => 1,..|V|-1 => |V|] codant la plage [[V]].
     * @see la section IMPLEMENTATION dans le DocBlock de ADigraph.
     * 
     * @return int[] Tableau indexé d'identifiants de noeuds.
     */
    abstract public function getNodeIds() : array ;

    /**
     * Retourne le tableau [0 => 1,..|E|-1 => |E|] codant la plage [[E]].
     * @see la section IMPLEMENTATION dans le DocBlock de ADigraph.
     * 
     * @return int[] Tableau indexé d'identifiants d'arcs.
     */
    abstract public function getArcIds() : array ;

    /**
     * Retourne l'encodage de f.
     * @see la section IMPLEMENTATION dans le DocBlock de l'ADigraph.
     * 
     * @return array[] tableau indexé de tableaux encodant f.
     */
    abstract public function getArcs() : array ;

    /**
     * Retourne f($arcId) où $arcId est l'identifiant d'un arc.
     * @see la section IMPLEMENTATION dans le DocBlock de ADigraph.
     * 
     * @param int $arcId L'identifiant de l'arc.
     * @throws InvalidArgumentException si $arcId n'est pas dans [[E]].
     * @return int[] Tableau d'entiers indexé sur 0..1 encodant f($arcId).
     */
    abstract public function getArc(int $arcId) : array ;

    /**
     * Retourne la restriction de f aux arcs entrants du noeud $nodeId.
     * L'extraction préserve les index des arcs dans f et leur ordre.
     * @see la section IMPLEMENTATION dans le DocBlock de l'ADigraph.
     * 
     * @param int $nodeId L'identifiant du noeud.
     * @throws InvalidArgumentException si $nodeId n'est pas dans [[V]].
     * @return array[] Tableau indexé de tableaux codant les arcs entrants du noeud.
     */
    abstract public function getIncomingArcs(int $nodeId) : array ;

    /**
     * Retourne la restriction de f aux arcs sortants du noeud $nodeId.
     * L'extraction préserve les index des arcs dans f et leur ordre.
     * @see la section IMPLEMENTATION dans le DocBlock de l'ADigraph.
     * 
     * @param int $nodeId L'identifiant du noeud.
     * @throws InvalidArgumentException si $nodeId n'est pas dans [[V]].
     * @return array[] Tableau indexé de tableaux codant les arcs sortants du noeud.
     */
    abstract public function getOutgoingArcs(int $nodeId) : array ;

    /**
     * Retourne le tableau des prédécesseurs directs du nœud $nodeId triés par ordre croissant.
     * 
     * @param int $nodeId L'identifiant du nœud.
     * @throws InvalidArgumentException si $nodeId n'est pas dans [[V]].
     * @return int[] Tableau indexé de nœuds triés par ordre croissant.
     */
    abstract public function getDirectPredecessors(int $nodeId) : array ;

    /**
     * Retourne le tableau des successeurs directs du nœud $nodeId triés par ordre croissant.
     * 
     * @param int $nodeId L'identifiant du nœud.
     * @throws InvalidArgumentException si $nodeId n'est pas dans [[V]].
     * @return int[] Tableau indexé de nœuds triés par ordre croissant.
     */
    abstract public function getDirectSuccessors(int $nodeId) : array ;

    /**
     * Retourne le tableau des nœuds atteignables à partir du nœud $nodeId, triés par ordre croissant.
     * 
     * @param int $nodeId L'identifiant du nœud.
     * @throws InvalidArgumentException si $nodeId n'est pas dans [[V]].
     * @return int[] Tableau indexé de nœuds triés par ordre croissant.
     */
    abstract public function getReachableNodes(int $nodeId) : array ;

    /**
     * Retourne le tableau des nœuds sources de G triés par ordre croissant.
     * 
     * @return int[] Tableau indexé de nœuds triés par ordre croissant.
     */
    abstract public function getSourceNodes() : array ;

    /**
     * Retourne le tableau des nœuds sources dans G, triés par ordre croissant.
     * 
     * @return int[] Tableau indexé de nœuds triés par ordre croissant.
     */
    abstract public function getSinkNodes() : array ;


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
    abstract public function getDipaths(int $origin, int $destination, bool $simple=false) : array ;

    /**
     * Retourne le tableau de tous les chemins du graphe si $simple est faux,
     * sinon renvoie le tableau de tous les chemins simples du graphe.
     * @voir la section IMPLEMENTATION dans le DocBlock de ADigraph.
     * 
     * @param bool $simple Indique s'il faut extraire tous les chemins simples (true) ou tous les chemins (false).
     * @return array[] Tableau indexé de tableaux codant tous les chemins dans G (ou tous les chemins simples dans G).
     */
    abstract public function getAllPaths(bool $simple=false) : array ;

    /**
     * Retourne le degré sortant du graphe.
     * 
     * @return int Le degré de sortie maximal des nœuds.
     */
    abstract public function getMaxOutDegree() : int ;

    /**
     * Retourne vrai si G est un digraphe simple.
     * 
     * @return bool
     */
    abstract public function isSimpleDigraph() : bool ;

    /**
     * Retourne vrai si G est cyclique.
     * 
     * @return bool Si G est cyclique (true) ou non (false).
     */
    abstract public function isCyclic() : bool ;

    /**
     * Retourne vrai si G est fortement connecté.
     * 
     * @return bool Si G est fortement connecté (true) ou non (false).
     */
    abstract public function isStronglyConnected() : bool ;

    /**
     * Construit et retourne un objet implémentant ADigraph à partir du fichier JSON $jsonFile.
     * 
     * @return ADigraph
     */
    static abstract public function fromJson(string $jsonFile) : ADigraph ;

    /**
     * Encode et retourne G au format JSON.
     * 
     * @return string
     */
    abstract public function toJson() : string ;

    /**
     * Encode et retourne les indicateurs de base de G sous forme de chaîne de caractères et si $showName est vrai, encode également le nom du graphe.
     * 
     * @param bool $showName Encode le nom du graphe (true) ou non (false)
     * @return string
     */
    abstract public function coreToString(bool $showName=false) : string ;

    /**
     * Encode et retourne le tableau indexé d'arcs $arcs sous forme de chaîne de caractères.
     * 
     * @param array[] $arcs Tableau indexé de tableaux.
     * @return string
     */
    abstract public function arcsToString(array $arcs) : string ;

    /**
     * Encode et retourne le tableau de chemins $paths sous forme de chaîne de caractères.
     * 
     * @param array[] $paths Tableau indexé de tableaux de tableaux.
     * @return string
     */
    abstract public function pathsToString(array $paths) : string ;

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
    abstract public function propertiesToString(bool $showName=false, bool $showProperties=false, bool $showDipaths=false, bool $showSimpleDipaths=false) : string;


    /**
     * Encode et retourne les éléments clés et les propriétés de G sous forme de chaîne de caractères.
     * 
     * @return string
     */
    abstract public function __toString() : string ;
}
?>