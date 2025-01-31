import {
    parseCSV,
    p_fetch
}
from "./utils.js";

/* 
Fichier CSV listant les régions françaises.
Définition des champs :
    https://www.insee.fr/fr/information/3363419#titre-bloc-26
*/
const url = "./data/régions.csv";

/* 
Promesse extrayant par Fetch le contenu du fichier CSV accessible à l'URL `url` 
et le renvoyant sous la forme d'un tableau d'objets représentant les régions.
Chaque objet a les propriétés suivantes :
- "code" : le numéro de la région (entier)
- "nom" : le nom de la région (chaîne).
Les objets sont triés dans l'ordre croissant des numéros de régions.
*/
export const p_régions =
    p_fetch(url, "csv")
    .then(csv => parseCSV(csv, "\r\n", ";"))
    .then(tab_régions => {
        let régions = [];
        // filtrage des champs et renommage
        tab_régions.forEach(région => {
            régions.push({
                "code": parseInt(région["REGION"]),
                "nom": région["NCCENR"]
            });
        });
        régions.sort((r1, r2) => r1.code > r2.code);
        console.debug("régions", régions);
        return régions;
    }).catch(erreur => {
        console.error(erreur.message);
        return [];
    });