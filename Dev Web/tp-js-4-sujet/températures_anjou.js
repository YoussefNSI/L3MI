import { p_températures_départementales } from "./températures_départementales.js";

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
export const p_températures_anjou = new Promise((resolve) => {
  var tempAnjou = p_températures_départementales[0];
  resolve(tempAnjou);
});

console.log(p_températures_départementales);
