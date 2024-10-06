import {
    p_régions
}
from "./régions.js";

import {
    p_départements
}
from "./départements.js";

/*
Le script communes.php fournit la liste des communes d'une région
en paginant les résultats par blocs successifs de 100 communes.
Le script répond aux requêtes HTTP POST en acceptant deux paramètres :
- "région" : le code de la région demandée
- "bloc" : le numéro du bloc de communes demandé.

Le script renvoie le bloc de communes sous la forme d'un tableau d'objets
JSON où chaque objet a 3 propriétés :
- "nom" : le nom de la commune (chaîne)
- "département" : le code du département de la commmune (chaîne)
- "région" : le code de la région (chaîne).

Le script renvoie un tableau vide si le bloc demandé est vide.

La fonction `extraireCommunes` ajoute au tableau `communes` de la région `région`
toutes les communes figurant dans les blocs de rang >= `bloc`.
Les blocs (tableaux) sont simplement concaténés sans transformation. 
La fonction est récursive et requête le script PHP à chaque appel 
en renvoyant la promesse Fetch correspondante.
*/
function extraireCommunes(communes, région, bloc) {
    // A REMPLACER
    return Promise.reject(new Error("extraireCommunes() à implémenter"));
}

/* 
Fonction prenant en paramètre le code d'une région (chaîne)
et renvoyant une promesse construite à partir des 3 promesses extrayant
- les départements du fichier XML (voir `p_départements` dans l'import).
- les régions du fichier CSV (voir `p_régions` dans l'import),
- les communes de la région en appelant `extraireCommunes`.

Les 3 promesses sont résolues en parallèle.
Si elles sont tenues, la réponse est un tableau d'objets
représentant toutes les communes de la région.
Chaque objet a 3 propriétés :
- "nom" : le nom de la commune (chaîne)
- "département" : le nom du département de la commmune (chaîne)
- "région" : le nom de la région (chaîne).

La transformation opérée consiste à nommmer départements et régions
qui ne sont que codifés par `extraireCommunes`.
*/
export function p_communes(région) {
    let communes = [];
    let bloc = 1;
    return Promise.all([p_régions, p_départements, extraireCommunes(communes, région, bloc)])
        .then(rég_dép_com => {
            let régions = rég_dép_com[0];
            let départements = rég_dép_com[1];
            let communes_région = rég_dép_com[2];
            communes_région = communes_région.map(commune => {
                commune["région"] = régions.find((région) => région.code == commune.région).nom;
                commune["département"] = départements[commune.département].nom;
                return commune;
            });
            console.debug("communes_région", communes_région);
            return communes_région;
        }).catch(erreur => {
            console.error(erreur.message);
        });
};