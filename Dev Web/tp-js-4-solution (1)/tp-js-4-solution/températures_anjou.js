import {
    p_températures_départementales
}
from "./températures_départementales.js";

/* 
Promesse produisant les températures minimales/moyennes/maximales mensuelles en 2022
dans le département 49 à partir de la réponse de la promesse `p_températures_départementales` (voir import).
La réponse est un tableau d'objets.
A chacun des 12 mois de 2022 correspond 3 objets dans ce tableau : 
- un pour la température minimale, 
- un pour la température maximale et
- un pour la température moyenne.

Chaque objet à 4 propriétés :
    -- "date" : un objet Date correspondant au 1er jour du mois
    -- "mois" : le nom du mois en français
    -- "température" : un flottant représentant une température
    -- "classe" : l'une des chaînes "Minimales", "Maximales", "Moyennes"

La température minimale (resp. maximale) d'un mois donné sera le minimum (resp. maximum) 
des températures minimales (resp. maximales) quotidiennes relevées pour ce mois.
La température moyenne d'un mois sera la moyenne des températures moyennes 
quotidiennes relevées pour ce mois.
*/
export const p_températures_anjou =
    p_températures_départementales.then(relevés_4D => {
        let relevés_anjou = relevés_4D["49"]
            .filter(relevé => (new Date(relevé.date)).getFullYear() === 2022);
        let relevés_mensuels_anjou = [];
        for (let i = 0; i < 12; ++i) {
            let relevés_mois = relevés_anjou.filter(relevé => (new Date(relevé.date)).getMonth() === i);
            let tmin = Math.min(...relevés_mois.map(relevé => relevé.tmin));
            let tmax = Math.max(...relevés_mois.map(relevé => relevé.tmax));
            let tmoy = relevés_mois.map(relevé => relevé.tmoy).reduce((accumulator, currentValue) => accumulator + currentValue, 0) / relevés_mois.length;
            let date = new Date(`2022-${i+1}-1`);
            let mois = new Intl.DateTimeFormat("fr-FR", {
                month: "long"
            }).format(date);
            relevés_mensuels_anjou = relevés_mensuels_anjou.concat([{
                    "température": tmin,
                    "classe": "Minimales",
                    "date": date,
                    "mois": mois
                },
                {
                    "température": tmoy,
                    "classe": "Moyennes",
                    "date": date,
                    "mois": mois
                },
                {
                    "température": tmax,
                    "classe": "Maximales",
                    "date": date,
                    "mois": mois
                }
            ]);
        }

        console.debug("relevés_mensuels_anjou", relevés_mensuels_anjou);
        return relevés_mensuels_anjou;
    }).catch(erreur => {
        console.error(erreur.message);
    });