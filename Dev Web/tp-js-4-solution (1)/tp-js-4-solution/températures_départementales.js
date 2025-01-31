import {
    p_fetch
}
from "./utils.js";

import {
    p_départements
}
from "./départements.js";

/* 
Fichier JSON contenant les relevés quotidiens de températures minimales, moyennes et maximales
pour les départements 49, 59, 65 et 83 sur la période 2018-2023.
Source : https://odre.opendatasoft.com/explore/dataset/temperature-quotidienne-departementale/information/?disjunctive.departement
*/
const url = "./data/49_59_65_83.json";

/* 
Promesse construite à partir des 2 promesses extrayant
- les relevés du fichier JSON  accessible à l'URL `url` en utilisant `p_fetch` (voir import),
- les départements du fichier XML (voir `p_départements` dans l'import).

Les 2 promesses sont résolues en parallèle.
Si elles sont toutes deux tenues, la réponse est un objet classant les relevés par département,
cad. dont les propriétés ont pour clé les numéros des départements (chaînes).
Pour chaque département, la propriété est le tableau d'objets correspondant à ses relevés.
Chaque objet représentant un relevé possède les propriétés :
- "tmin" : la température minimale (flottant) du relevé,
- "tmoy" : la température moyenne (flottant) du relevé,
- "tmax" : la température maximale (flottant) du relevé,
- "code" : le numéro (chaîne) du département pour ce relevé,
- "date" : l'objet JS `Date` correspondant à la date du relevé.
*/
export const p_températures_départementales =
    Promise.all([p_départements, p_fetch(url, "json")])
    .then(résultats => {
        // départements
        const départements = résultats[0];
        // JSON des relevés
        const relevés = résultats[1];
        // classification des relevés par département
        let relevés_départements = {};
        // conversion des dates en objet Date et ajout du nom du département
        relevés.forEach(relevé => {
            let code = relevé.code;
            let relevé1 = relevé;
            if (relevés_départements[code]) {
                relevés_départements[code].push(relevé1);
            } else {
                relevés_départements[code] = [relevé1];
            }
            // extraction et ajout du nom du département
            relevé1.département = départements[code].nom;
            // conversion des dates en objet Date
            /* !! Attention date1!=date2
            let date1 = new Date("2022-12-1");
            let date2 = new Date("1/12/2022");
            console.log("2022-12-1", date1)
            console.log("1/12/2022", date2)
            /** */
            relevé1.date = new Date(relevé1.date);
        });
        console.debug("relevés_départements", relevés_départements);
        return relevés_départements;
    }).catch(erreur => {
        console.error(erreur.message);
    });