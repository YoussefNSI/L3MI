import {
    p_fetch
}
from "./utils.js";

import {
    p_régions
}
from "./régions.js";

/* 
Fichier XML listant les départements français.
Définition des balises :
    https://www.insee.fr/fr/information/3363419#titre-bloc-23
*/
const url = "./data/départements.xml";

/* 
Promesse construite à partir des 2 promesses extrayant
- le contenu du fichier XML accessible à l'URL `url` en utilisant `p_fetch` (voir import),
- les régions du fichier CSV (voir `p_régions` dans l'import).

Les promesses sont résolues en parallèle.
Si elles sont toutes deux tenues, la réponse est un objet dont les propriétés sont 
des objets correspondant aux éléments `département` du document XML.
La clé d'une propriété est la chaîne correspondant au code du département (p. ex. 
"2A", "49").
Chaque objet a les propriétés suivantes
- "nom" : contenu du sous-élément NCCENR (chaîne)
- "chef_lieu" : contenu du sous-élément CHEFLIEU (entier)
- "région" : nom de la région du département (chaîne).
*/
export const p_départements =
    Promise.all([p_fetch(url, "xml"), p_régions]).then(([dataXML, regions]) => {
        const parser = new DOMParser();
        const doc = parser.parseFromString(dataXML, "application/xml");
        const départements = [];
        doc.documentElement.querySelectorAll("département").forEach(département => { 
            départements.push({
                "nom": département.querySelector("NCCENR").textContent,
                "chef_lieu": département.querySelector("CHEFLIEU").textContent,
                "région": regions.find(region => region["REGION"] === département.querySelector("REGION").textContent)["NCC"]
            });
        });
        return départements;
        });
            
    Promise.resolve(Object.fromEntries(((new Array(100)).fill(0)).map((v, k) => [k, {
        "nom": ["M&L", "N", "HP", "V"][Math.floor(4 * Math.random())],
        "chef_lieu": "y",
        "région": "z"
    }])));