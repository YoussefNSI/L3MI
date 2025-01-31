/*
Exemple de carte au format JSON.
*/
let map_3_3_acyclic_json = ' \
{ \
    "name":"map_3_3_acyclic", \
    "digraph": { \
        "nrNodes": 3, \
        "nrArcs": 3, \
        "arcs": [ \
            [1, 2], \
            [1, 3], \
            [2, 3] \
        ] \
    }, \
    "labeling": { \
        "concepts": ["c1", "c2", "c3"], \
        "influenceType": "symbolic", \
        "influences": [ \
            "+", \
            "+", \
            "-" \
        ] \
    } \
} \
';

/*
Conversion de carte JSON en objet JS à utiliser pour certains tests.
*/
let map_3_3_acyclic = JSON.parse(map_3_3_acyclic_json);


/*
QUESTION 1
La fonction influence(carte, i, j) prend comme paramètres
- carte : un objet JS modélisant une carte (voir format JSON)
- i : le numéro d'un noeud de la carte (>=1)
- j : le numéro d'un noeud de la carte (>=1).

Elle renvoie
- la chaîne vide si [i,j] ne figure pas dans le tableau de la sous-propriété "arcs" de la carte
- la k-ième chaîne du tableau de la sous-propriété "influences" de la carte si [i,j] est le k-ième élément du tableau "arcs".

Par exemple, la fonction renvoie la chaîne "-" pour la paire (2,3) de l'objet map_3_3_acyclic.
*/
function influence(carte, i, j) {
    // A COMPLETER
}


/*
QUESTION 2
La fonction champs(carte) prend comme paramètre un objet JS modélisant une carte (voir format JSON).

Elle doit, dans l'ordre,
- remplir le champ étiqueté par "noeuds" avec le nombre de noeuds de la carte
- remplir le champ étiqueté par "arcs" avec le nombre d'arcs de la carte
- désactiver ces champs
- griser leur couleur de fond
- contraindre les champs numériques étiquetés par "ORIGINE" et "DESTINATION" à ne pas dépasser le nombre de noeuds de la carte
- fixer l'attribut 'value' du premier à 1 et l'attribut 'value' du second au nombre de noeuds de la carte.
*/
function champs(carte) {
    // A COMPLETER
}

champs(map_3_3_acyclic);


/*
La fonction matrice(carte) prend comme paramètre un objet JS modélisant une carte (voir format JSON).

Elle affiche d'abord la matrice d'adjacence valuée de la carte sous forme de tableau HTML dans l'élément de classe "matrice"
(en remplaçant si nécessaire le contenu pré-existant) :
(1) ligne et colonne d'en-têtes affichent les concepts associés aux noeuds (sous-propriétés "concepts" de la carte)
(2) chaque cellule (i,j) correspondant à un arc [i,j] de la carte contient l'influence de cet arc (caractère "+"" ou "-")
(3) chaque cellule (i,j) pour laquelle l'arc [i,j] n'existe pas est laissée vide
(4) les deux cellules d'en-têtes correspondant aux numéros de noeuds renseignés dans les champs "ORIGINE" et "DESTINATION" 
prennent la couleur de fond de ces champs et sont réactualisés à chaque changement.

Attention : cette fonction fait appel à la fonction 'influence'.
L'appel suivant affiche la carte de l'objet map_3_3_acyclic au chargement (si la fonction 'influence
est correctement implémentée).
*/
matrice(map_3_3_acyclic);


/*
QUESTION 3
Implémentez un écouteur sur les boutons radio : 
- tout clic sur le bouton 'POSITIF' place les cellules de coefficient '+' dans la classe HTML 'positif' et retire les cellules de coefficient '-' de la classe HTML 'négatif' 
- tout clic sur le bouton 'NEGATIF' place les cellules de coefficient '-' dans la classe HTML 'négatif' et retire les cellules de coefficient '+' de la classe HTML 'positif'.
*/

// A COMPLETER


/*
QUESTION 4
La sélection d'une carte à l'aide du menu déroulant déclenche une requête Ajax récupérant le fichier JSON
de la carte par méthode HTTP GET :
- le nom du fichier correspond à l'attribut 'value' de l'option choisie, suffixé par ".json"
- le fichier se situe dans le dossier "./data".

Formulaire et tableau HTML doivent être actualisés par appel aux fonctions 'champs' et 'matrice'.
*/

// A COMPLETER


/*
QUESTION 5
Un clic sur le bouton 'chemins' affiche les chemins (simples) reliant les noeuds origine et destination renseignés dans les champs numériques.
Le calcul est délégué au script './php/api.php' auquel les champs suivants sont communiqués par requête HTTP POST :
- "carte" : le nom de la carte
- "origine" : le numéro du noeud origine
- "destination" : le numéro du cnoeud destination.

Les chemins sont renvoyés au format JSON sous forme de tableau de tableaux. Par exemple, la requête paramétrée par
carte=map_3_3_acyclic&origine=1&destination=3 renvoie le tableau [[1,2,3],[1,3]] des 2 chemins possibles. 
Les chemins récupérés et les messages d'erreurs seront simplement affichés en console.
*/

// A COMPLETER