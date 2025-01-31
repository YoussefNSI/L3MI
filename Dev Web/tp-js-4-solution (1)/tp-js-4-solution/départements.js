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
    Promise.all([p_régions, p_fetch(url, "xml")])
    .then(résultats => {
        let régions = résultats[0];
        let xml = résultats[1];

        // DOM du document XML
        let xml_doc = (new window.DOMParser()).parseFromString(xml, "text/xml");

        // extraction nom-code départements
        let départements = {};
        xml_doc.querySelectorAll("département").forEach(département => {
            let code = département.querySelector("DEP").textContent;
            let chef_lieu = parseInt(département.querySelector("CHEFLIEU").textContent);
            let nom = département.querySelector("NCCENR").textContent;
            let région = régions.filter(région => région.code === parseInt(département.querySelector("REGION").textContent))[0].nom;
            départements[code] = {
                "nom": nom,
                "chef_lieu": chef_lieu,
                "région": région
            };
        });
        console.log("départements", départements);
        return départements;
    }).catch(erreur => {
        console.error(erreur.message);
        return [];
    });